from __future__ import print_function

import sys
import time

import networkx as nx
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import objectives
from keras.layers import Dense, Dropout, Flatten, Input, Add, Merge, Average, BatchNormalization, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import RandomNormal
from keras.utils import to_categorical
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy.sparse import csgraph
from custom_losses import crossentropy_weighted
from feature_eng import *
from layers.graph import GraphConvolution, GraphConvolutionReprise
from utils import *
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA, PCA

# Define parameters
DATASET = 'cora'  # 'citeseer' 'pubmed'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 1000  # early stopping patience

# Get data
A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    DATASET)

# Normalize X
# X_enh = enhance_features(A.A, X.A)

# X = pca(X, 1433)
X /= X.sum(1).reshape(-1, 1)
A = (np.logical_or(A.A, A.transpose().A))
A = csr_matrix(A)

A_pow = adj_pow(A)


def get_pred_label_diffusion(A_pow, labels, mask):
    c_labels = decode_onehot(labels, mask)
    n_labels = np.max(c_labels)

    c_labels_pred = np.zeros((A_pow.shape[0], n_labels))

    for i in range(A_pow.shape[0]):
        c_labels_pred[i] = [A_pow[i].dot(c_labels == lab)
                            for lab in range(1, n_labels+1)]

    return c_labels_pred


# from sklearn.semi_supervised import LabelSpreading

# y_for_sklearn = np.ones(A.shape[0]) * -1
# y_for_sklearn[train_mask] = np.argmax(y_train[train_mask], 1)
# label_spreading = LabelSpreading(kernel=(lambda x, y: A)).fit(X, y_for_sklearn)
# y_base = label_spreading.predict(X)


y_base = get_pred_label_diffusion(A_pow, y_train, train_mask)

y_base_cat = to_categorical(
    np.argmax(y_base, axis=1), num_classes=y_base.shape[1])

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)

    support = 1
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = [X]+T_k
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)
         for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

X_in = Input(shape=(X.shape[1],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.


def model_kipf():
    # graph[0] = np.random.normal(size=graph[0].shape)
    H = Dropout(0.8)(X_in)
    H = GraphConvolution(32, support, activation='linear',
                         kernel_regularizer=l2(5e-5))([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y_train.shape[1], support, kernel_regularizer=l2(l=0.01),
                         activation='softmax', name='y')([H]+G)
    # Compile model
    model = Model(inputs=[X_in]+G, outputs=Y)
    model.compile(loss=crossentropy_weighted(train_mask),
                  optimizer=Adam(lr=0.01), metrics=['accuracy'])
    return model


def model_label_reconstruction():
    H = Dropout(0.6)(X_in)
    H = GraphConvolution(32, support, activation='linear',
                         kernel_regularizer=l2(5e-5))([H]+G)
    H = Dropout(0.6)(H)
    Y = GraphConvolution(y_train.shape[1], support, kernel_regularizer=l2(l=0.01),
                         activation='softmax', name='y')([H]+G)
    Z = GraphConvolution(y_train.shape[1], support, kernel_regularizer=l2(l=0.01),
                         activation='softmax', name='z')([H]+G)

    # Compile model
    model = Model(inputs=[X_in]+G, outputs=[Y, Z])

    model.compile(loss={'z': crossentropy_weighted(~train_mask), 'y': crossentropy_weighted(train_mask)},
                  optimizer=Adam(lr=0.01), metrics=['accuracy'])

    return model


def model_graph_reconstruction():

    H = Dropout(0.5)(X_in)
    H = GraphConvolution(64, support, activation='linear',
                         kernel_regularizer=l2(5e-5))([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y_train.shape[1], support, kernel_regularizer=l2(l=0.01),
                         activation='softmax', name='y')([H]+G)
    Z = GraphConvolution(A.shape[1], support, kernel_regularizer=l2(l=0.01),
                         activation='selu', trainable=True, name='z')([H]+G)

    # Compile model
    model = Model(inputs=[X_in]+G, outputs=[Y, Z])

    model.compile(loss={'z': 'kullback_leibler_divergence', 'y': crossentropy_weighted(train_mask, 5)},
                  optimizer=Adam(lr=0.01), metrics=['accuracy'])

    return model


def model_baseline():

    X_in = Input(shape=(graph[0].shape[1],))
    H = Dense(64, activation='linear')(X_in)
    H = Dropout(0.5)(H)
    H = Dense(64, activation='linear', kernel_regularizer='l2')(H)
    H = Dropout(0.5)(H)
    Y = Dense(y_train.shape[1], activation='softmax')(H)
    model = Model(inputs=[X_in] + G, outputs=Y)

    model.compile(loss=crossentropy_weighted(train_mask),
                  optimizer=Adam(lr=0.01), metrics=['accuracy'])
    return model


def model_baseline_preproc_step():
    alpha = 0.55
    A_norm = A / A.sum(0)
    graph[0] = alpha * A_norm.dot(graph[0])
    #  + \
    #     (1 - alpha) * A_norm.dot(A_norm.dot(graph[0]))

    # graph[0] = PCA(n_components=50).fit_transform(graph[0])
    return model_baseline()


def model_baseline_preproc_conv():
    graph[0] = csgraph.laplacian(A, normed=True).dot(X)

    return model_baseline()


def model_crazy_graph_idea():
    graph.append(y_train)

    L = [Input(shape=(None, None), batch_shape=(None, None), sparse=False)]
    H = Dropout(0.5)(X_in)
    H = GraphConvolutionReprise(128, n_classes=7, support=support, activation='linear',
                                kernel_regularizer=l2(4e-5))([H]+G+L)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y_train.shape[1], support, kernel_regularizer=l2(l=0.01),
                         activation='softmax', name='y')([H]+G)

    # Compile model
    model = Model(inputs=[X_in]+G+L, outputs=[Y])

    model.compile(loss={'y': crossentropy_weighted(train_mask, 5)},
                  optimizer=Adam(lr=0.01), metrics=['accuracy'])

    return model


def model_crazy_idea_reprise():
    graph.append(y_train)
    # X_in = Input(shape=(50,))

    # kpca = KernelPCA(kernel="rbf", gamma=7, n_components=50)
    # graph[0] = kpca.fit_transform(graph[0])
    L = Input(shape=(None, 7),
              batch_shape=(None, 7), sparse=False)


# 32 8 32 drop - 80
# 32 8 8 32 drop - 80
#

    H = Dropout(0.5)(X_in)
    J = Dropout(0.5)(L)
    H = GraphConvolution(units=32, support=support, kernel_regularizer=l2(
        5e-5), activation='linear')([H]+G)
    H = Dropout(0.5)(H)
    J = GraphConvolution(units=32, support=support, kernel_regularizer=l2(
        5e-5), activation='relu')([J]+G)
    J = GraphConvolution(units=16, support=support, activation='relu')([J]+G)
    J = GraphConvolution(units=16, support=support, activation='relu')([J]+G)
    J = GraphConvolution(units=16, support=support,
                         activation='linear', kernel_regularizer=l2(5e-5))([J]+G)
    H = Concatenate()([H, J])
    H = Dropout(0.5)(H)
    H = GraphConvolution(units=32, support=support, kernel_regularizer=l2(
        5e-5), activation='linear')([H]+G)
    Y = GraphConvolution(y_train.shape[1], support, kernel_regularizer=l2(l=0.01),
                         activation='softmax', name='y')([H]+G)

    model = Model(inputs=[X_in]+G+[L], outputs=[Y])

    model.compile(loss={'y': crossentropy_weighted(train_mask, 5)},
                  optimizer=Adam(lr=0.01), metrics=['accuracy'])

    return model


def model_concatenate_features():
    # X_in = Input(shape=(X.shape[1],))  # + y_train.shape[1],))

    # y train graph average
    # A_loop = A + np.eye(A.shape[0])
    # y_attempt = A.dot(A.dot(y_train))

    # y_attempt_c = np.argmax(y_attempt, axis=1)
    # y_attempt_c = to_categorical(y_attempt_c, num_classes=7)

    # mask = np.max(y_attempt, axis=1) == 0
    # mask = mask.tolist()
    # for n, m in enumerate(mask):
    #     if m is True:
    #         y_attempt_c[n, :] = 0

    # y_attempt_c[train_mask] = y_train[train_mask]
    y_attempt = y_train
    # y_attempt = y_train + np.random.normal(loc=1, size=y_train.shape)
    # graph[0] = graph[0] + np.random.normal(loc=0, size=graph[0].shape)

    graph[0] = np.concatenate((graph[0], y_attempt), axis=1)

    # graph[0] = np.random.normal(loc=0, size=graph[0].shape)
    # graph[0] = np.random.choice(
    #     a=[0, 1], size=graph[0].shape, p=[0.8, 0.2])

    # graph[0] = PCA(n_components=200).fit_transform(graph[0])

    # kpca = KernelPCA(kernel="rbf", gamma=30, n_components=200)
    # graph[0] = kpca.fit_transform(graph[0])

    X_in = Input(shape=(graph[0].shape[1],))
    H = Dropout(0.8)(X_in)
    H = GraphConvolution(64, support, activation='linear',
                         kernel_regularizer=l2(5e-5))([H]+G)
    H = Dropout(0.5)(H)
    H = GraphConvolution(64, support, activation='linear',
                         kernel_regularizer=l2(5e-5))([H]+G)
    H = Dropout(0.5)(H)
    H = GraphConvolution(64, support, activation='linear',
                         kernel_regularizer=l2(5e-5))([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y_train.shape[1], support, kernel_regularizer=l2(l=0.01),
                         activation='softmax', name='y')([H]+G)
    # Compile model
    model = Model(inputs=[X_in]+G, outputs=Y)
    model.compile(loss=crossentropy_weighted(train_mask),
                  optimizer=Adam(lr=0.01), metrics=['accuracy'])
    return model


def model_svm():
    alpha = 0.55
    A_norm = (A + np.eye(N=A.shape[0])) / A.sum(0)
    graph[0] = graph[0] + A.dot(graph[0]) + A.dot(A.dot(graph[0]))
    #  + \
    #     (1 - alpha) * A_norm.dot(A_norm.dot(graph[0]))

    # graph[0] = PCA(n_components=50).fit_transform(graph[0])

    def one_hot_to_cat(x):
        return np.argmax(x, axis=1)

    model = SVC(decision_function_shape='ovo', C=1, kernel='rbf')

    model.fit(graph[0][train_mask, :], one_hot_to_cat(y_train[train_mask]))
    print(model.score(graph[0][test_mask],
                      one_hot_to_cat(y_test[test_mask, :])))
    print(model.score(graph[0][val_mask],
                      one_hot_to_cat(y_val[val_mask, :])))

    sys.exit(0)


# model = model_graph_reconstruction()
# model = model_label_reconstruction()
# model = model_kipf()
# model = model_baseline()
# model = model_baseline_preproc_step()
# model= model_crazy_graph_idea()
model = model_crazy_idea_reprise()
# model = model_concatenate_features()
# model = model_svm()


def training():
    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999

    # Fit
    # model.fit(x=[X, A_], y=y_train, batch_size=A.shape[0], sample_weight=train_mask,
    #           epochs=200, shuffle=False)

    # model.fit(x=[X, A_], y=[y_train, y_base_cat],
    #           batch_size=A.shape[0], epochs=500, shuffle=False)

    for epoch in range(1, NB_EPOCH+1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        # model.fit(graph, y={'y': y_train, 'z': A.A},  # sample_weight=train_mask,
        #           batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        model.fit(graph, y=y_train,  # sample_weight=train_mask,
                  batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [train_mask, val_mask])
        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))

        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1
    return model, preds


def validating(preds):
    val_loss, val_acc = evaluate_preds(preds, [y_val], [val_mask])
    print("Val set results:",
          "loss= {:.4f}".format(val_loss[0]),
          "accuracy= {:.4f}".format(val_acc[0]))
    return val_acc[0]


def testing(preds):
    # Testing
    test_loss, test_acc = evaluate_preds(preds, [y_test], [test_mask])
    print("Test set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))
    return test_acc[0]


model, preds = training()
results_test = testing(preds)
results_val = validating(preds)


def crossvalid(get_model, params):
    results = []
    for n, param in enumerate(params):
        model = get_model(**param)
        training()
        results.append(testing())

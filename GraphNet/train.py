from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                          Embedding, Flatten, Input, MaxPool2D)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1_l2

import keras.metrics

import numpy as np
from data.utils import load_data
from gcnlayer import GraphConvolution
from feature_eng import enhance_features, normalize
from utils import preprocess_adj
from custom_losses import crossentropy_weighted


from scipy.sparse.csgraph import laplacian as scipy_laplacian
N_EPOCHS = 1000

A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    'cora')
X = X.A

n_features = X.shape[1]
n_vertex = A.shape[0]


A_ = preprocess_adj(A)
graph = [X, A_]


def get_model_kipf():
    adj = Input(shape=(None, None), batch_shape=(None, None), sparse=False)
    inp = Input(shape=(n_features,))

    H = Dropout(0.5)(inp)
    H = GraphConvolution(16, kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                         activation='relu',)([H] + [adj])
    H = Dropout(0.5)(H)
    Y = GraphConvolution(
        y_train.shape[1], activation='softmax')([H] + [adj])

    model = Model(inputs=[inp, adj], outputs=Y)

    model.compile(loss=crossentropy_weighted(train_mask),
                  optimizer=Adam(5e-5), metrics=['accuracy'], )

    return model


X /= X.sum(axis=1).reshape(-1, 1)


def accuracy(predicted, labels):
    matching = np.argmax(predicted, axis=1) == np.argmax(labels, axis=1)
    return np.mean(matching)


model = get_model_kipf()
# training standard algorithm:
model.fit(x=graph, y=y_train, batch_size=n_vertex,
          epochs=N_EPOCHS, verbose=1, shuffle=False)

# for epoch in range(1, N_EPOCHS + 1):

#     model.fit(x=graph, y=y_train,
#               sample_weight=train_mask,
#               batch_size=n_vertex,
#               epochs=1, verbose=0,
#               shuffle=False,
#               )

#     pred = model.predict([features, A], batch_size=n_vertex)

#     train_acc = accuracy(pred[train_mask], y_train[train_mask])
#     test_acc = accuracy(pred[test_mask], y_test[test_mask])

#     print("Epoch: {}/{}".format(epoch, N_EPOCHS),
#           "train_acc: {:.4f}".format(train_acc),
#           "test_acc: {:.4f}".format(test_acc))

pred = model.predict(graph, batch_size=n_vertex)
test_acc = accuracy(pred[test_mask], y_test[test_mask])

print('Test accuracy: ', test_acc)

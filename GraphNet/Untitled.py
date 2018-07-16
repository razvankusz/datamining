
# coding: utf-8

# In[2]:


from gcn.utils import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse import csgraph
from tqdm import tqdm

from sklearn.svm import SVC


# In[3]:


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
features = features.A


# In[4]:


def one_hot_to_cat(X):
    '''
    shape of X = (n_samples, n_classes)
    '''
    return np.apply_along_axis(arr=X, axis=1, func1d=lambda x: np.argmax(x))


# In[34]:


model = SVC(decision_function_shape='ovr', C=1, kernel='linear')

def try_model(features, train_mask, y_train, model):
    model.fit(features[train_mask, :], one_hot_to_cat(y_train[train_mask]))
    print model.score(features[test_mask], one_hot_to_cat(y_test[test_mask,:]))


# In[35]:


def adj_to_graph(adj):
    A = coo_matrix(adj)    
    N, _ = adj.shape
    edges = zip(A.row, A.col)
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    graph.add_edges_from(edges)
    return graph


# In[36]:


graph = adj_to_graph(adj)


# In[54]:


# train by stacking features of neighbours
feature_extra = np.zeros(features.shape)

for n in graph.nodes():
    feature_neighbors = np.mean([features[m] for m in graph.neighbors(n)], axis=0)
    
    feature_neighbors_2 = []
    for m in graph.neighbors(n):
        feature_neighbors_2.append(np.mean([features[p] for p in graph.neighbors(m)], axis=0))
    feature_neighbors_2 = np.mean(feature_neighbors_2, axis=0)
    
    feature_extra[n] = 0 * features[n] + 3 * feature_neighbors + 5 * feature_neighbors_2 


# In[55]:


try_model(feature_extra, train_mask, y_train, model)


# In[10]:


def concat_features(features, n, params):
    padding = params.get('padding', 10)
    n_features = len(features[0])
    
    feature = []
    feature.append(features[n])
    feature = feature + [features[m] for m in graph.neighbors(n)]    

    if len(feature) >= padding:
        feature = feature[:padding]
    else:
        feature = feature + [np.zeros(n_features) for _ in range(padding - len(feature))]
    
    
    assert len(feature) == padding
    return np.array(feature)
    
def apply_feature_transformation(features, func, params={}):
    new_features = []
    N = len(features)
    
    for n in range(N):
        new_features.append(func(features, n, params))
    
    return np.array(new_features)

features_neigh = apply_feature_transformation(features, concat_features)


# In[14]:


from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization, Embedding
from keras import regularizers, losses

n_epoch = 10
batch_size= 20

def get_model(n_features, n_classes):
    # Model parameters
    input_shape = n_features

    inp = Input(shape=input_shape)

    flat = Flatten()(inp)
    hidden_1 = Dense(2048, activation='relu')(flat)
    dropout_1 = Dropout(0.2)(hidden_1)
    
    hidden_2 = Dense(1024, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.4)(hidden_2)
    
    hidden_3 = Dense(512, activation='relu')(dropout_2)
    hidden_4 = Dense(512, activation='tanh')(hidden_3)
    dropout_3 = Dropout(0.2)(hidden_4)
    
    out = Dense(n_classes, activation='softmax')(dropout_3)

    model = Model(inputs=inp, outputs=out)

    print(model.summary())
    return model

n_features = len(features[0])
n_classes = len(y_train[0])

features_neigh = apply_feature_transformation(features, concat_features, params={'padding': 10})

model = get_model((10, n_features), n_classes)

def train_score_model(model, features, y_train, y_test):
    model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
    
    model.fit(features[train_mask, :], y_train[train_mask, :], batch_size=batch_size, epochs=n_epoch,
          verbose=1, validation_data=(features[test_mask], y_test[test_mask]))
    print model.evaluate(features[test_mask], y_test[test_mask], batch_size=batch_size)
    
train_score_model(model, features_neigh, y_train, y_test)


# In[15]:


adj_pow = adj.copy()

for _ in range(4):
    adj_pow = adj.dot(adj_pow)
adj_pow = adj_pow / adj_pow.sum(axis=0)


# In[13]:


centrality = np.array([len(graph.neighbors(n)) for n in graph.nodes()])

y_cat_train = np.argmax(y_train, axis=1)


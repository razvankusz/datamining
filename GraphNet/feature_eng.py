from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import laplacian
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA

PCA_N_COMPONENTS = 40


def pca(X, n_components=PCA_N_COMPONENTS):
    n_samples, n_features = X.shape
    pca_model = PCA(n_components=n_components)
    return pca_model.fit_transform(X)


def laplacian_prep(A):
    L = laplacian(A, normed=True) - np.eye(A.shape[0])
    return csr_matrix(L)


def adj_to_graph(adj):
    A = coo_matrix(adj)
    N, _ = adj.shape
    edges = zip(A.row, A.col)
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    graph.add_edges_from(edges)
    return graph


def enhance_features(A, X):
    '''returns enhanced X'''

    graph_repr = adj_to_graph(A)
    feature_extra = np.zeros(X.shape)

    for n in graph_repr.nodes():
        feature_neighbors = np.mean([X[m]
                                     for m in graph_repr.neighbors(n)], axis=0)

        feature_neighbors_2 = []
        for m in graph_repr.neighbors(n):
            feature_neighbors_2.append(
                np.mean([X[p] for p in graph_repr.neighbors(m)], axis=0))
        feature_neighbors_2 = np.mean(feature_neighbors_2, axis=0)

        feature_extra[n] = 2 * feature_neighbors + feature_neighbors_2

    return feature_extra


def normalize(X):
    return X / X.sum(1).reshape(-1, 1)


def adj_pow(A, max_power=4):
    A1 = A + np.eye(A.shape[0])
    A_ = A1 / np.sum(A1, axis=1).reshape(-1, 1)

    for n in range(max_power):
        A_ = A_.dot(A1)

    return A_

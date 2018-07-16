import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_eng import adj_to_graph
from utils import load_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
DATASET = 'cora'

A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    DATASET)

A = A.A
X = X.A
graph = adj_to_graph(A)


def plot_features():
    plt.imshow(X)
    plt.show()
    plt.savefig('./vis/sparse_features.eps', format='eps',
                dpi=1000, bbox_inches='tight')


X_pca = PCA(n_components=30).fit_transform(X)


def plot_features_pca():
    plt.imshow(X_pca)
    plt.savefig('./vis/pca100.png', dpi=1000)
    plt.show()


X_pca_tsne = TSNE(perplexity=30.0).fit_transform(X_pca)

X_pca_tsne_dict = {i: X_pca_tsne[i] for i in range(len(X_pca_tsne))}


def plot_features_tsne():
    plt.scatter(x=X_pca_tsne[:, 0], y=X_pca_tsne[:, 1], alpha=0.6)
    nx.draw_networkx_edges(graph, pos=X_pca_tsne_dict,
                           alpha=0.1, edge_color='b')
    plt.show()


plot_features_tsne()

'''
Here implement a function that takes feature matrix 'X' and true label 'y'
Runs kmeans with correct number of clusters
Computes:
    ACC (clusteting_accuracy)
    NMI (nmi_score)
Return both values
'''
import numpy as np

from sklearn.cluster import KMeans

from src.evaluation.metrics import clustering_accuracy, nmi_score

def evaluate_clustering(X, y):
    n_clusters = len(np.unique(y))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X)

    acc = clustering_accuracy(y, y_pred)
    nmi = nmi_score(y, y_pred)

    return acc, nmi
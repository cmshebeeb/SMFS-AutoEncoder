import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def compute_laplacian(X, sigma=1.0):
    """
    Compute a similarity-based graph Laplacian.

    Parameters
    ----------
    X : array-like
        Data matrix (samples × features).
    sigma : float
        Gaussian similarity bandwidth.

    Returns
    -------
    L : numpy.ndarray
        Graph Laplacian matrix.
    """

    distances = euclidean_distances(X)

    W = np.exp(-(distances ** 2) / (2 * sigma ** 2))

    D = np.diag(W.sum(axis=1))

    L = D - W

    return L
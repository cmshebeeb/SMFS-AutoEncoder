import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def compute_paper_laplacian(X, n_neighbors=5, sigma=None):
    """
    Compute the similarity-based graph Laplacian
    following the graph construction used in the paper.

    Parameters
    ----------
    X : array-like
        Data matrix with shape (n_samples, n_features).

    n_neighbors : int
        Number of nearest neighbours.

    sigma : float
        Gaussian similarity bandwidth.

    Returns
    -------
    L : numpy.ndarray
        Graph Laplacian with shape (n_samples, n_samples).
    """

    X = np.asarray(X)

    distances = euclidean_distances(X)

    if sigma is None:
        nonzero_distances = distances[
            distances > 0
        ]

        sigma = np.median(
            nonzero_distances
        )

        print(
            f"Using data-dependent sigma: {sigma:.6f}"
        )

    n_samples = X.shape[0]

    W = np.zeros(
        (n_samples, n_samples),
        dtype=np.float64
    )

    for i in range(n_samples):

        # Exclude the point itself
        neighbor_indices = np.argsort(
            distances[i]
        )[1:n_neighbors + 1]

        for j in neighbor_indices:

            similarity = np.exp(
                -(distances[i, j] ** 2)
                / (2 * sigma ** 2)
            )

            W[i, j] = similarity

    # Make graph symmetric
    W = np.maximum(W, W.T)

    # Degree matrix
    D = np.diag(W.sum(axis=1))

    # Unnormalized graph Laplacian
    L = D - W

    return L
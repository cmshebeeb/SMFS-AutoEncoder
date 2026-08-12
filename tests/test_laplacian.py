import numpy as np

from src.graph.laplacian import compute_laplacian


def test_laplacian_four_points():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [3.0, 3.0],
        [3.0, 4.0]
    ])

    L = compute_laplacian(X)

    print("\nLaplacian:\n", L)

    # Laplacian must be square
    assert L.shape == (4, 4)

    # Laplacian is symmetric
    assert np.allclose(L, L.T)

    # Row sums should be approximately zero
    assert np.allclose(L.sum(axis=1), 0)
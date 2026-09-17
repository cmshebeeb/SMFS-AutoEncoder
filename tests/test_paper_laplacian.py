import numpy as np

from src.graph.paper_laplacian import compute_paper_laplacian


def test_paper_laplacian():

    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    L = compute_paper_laplacian(
        X,
        n_neighbors=2,
        sigma=1.0
    )

    print("Paper Laplacian:")
    print(L)

    assert L.shape == (4, 4)
    assert np.isfinite(L).all()

    # Laplacian should be symmetric
    assert np.allclose(L, L.T)

    # Diagonal should be non-negative
    assert np.all(np.diag(L) >= 0)

    # Row sums should approximately equal zero
    assert np.allclose(
        L.sum(axis=1),
        0.0
    )
import torch

from src.graph.auxiliary_h import update_H


def test_update_H():

    L_X = torch.tensor([
        [2.0, -1.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -1.0, 2.0]
    ])

    H = torch.tensor([
        [1.0, 0.5, 0.2],
        [0.3, 1.0, 0.4],
        [0.1, 0.6, 1.0]
    ])

    H_new = update_H(H, L_X)

    print("Original H:")
    print(H)

    print("Updated H:")
    print(H_new)

    assert H_new.shape == H.shape
    assert torch.isfinite(H_new).all()
    assert not torch.allclose(H_new, H)
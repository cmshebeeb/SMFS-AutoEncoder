import torch

from src.graph.reconstructed_structure import reconstructed_structure_loss


def test_reconstructed_structure_loss():

    reconstructed = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 4.0],
        [3.0, 2.0, 1.0]
    ])

    H = torch.eye(3)

    loss = reconstructed_structure_loss(
        reconstructed,
        H
    )

    print("Reconstructed structure loss:", loss.item())

    assert loss.ndim == 0
    assert loss.item() > 0
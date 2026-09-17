import torch


def reconstructed_structure_loss(
    reconstructed,
    H
):
    """
    Paper reconstructed-structure term.

    Paper form:

        Tr(W2^T Z H H^T Z^T W2)

    With our PyTorch orientation:

        reconstructed = W2^T Z

    where:

        reconstructed : (n_features, n_samples)
        H             : (n_samples, n_samples)

    Therefore:

        reconstructed @ H

    has shape:

        (n_features, n_samples)

    and the trace term is equivalent to:

        || reconstructed @ H ||_F^2
    """

    structure = reconstructed @ H

    return torch.sum(structure ** 2)
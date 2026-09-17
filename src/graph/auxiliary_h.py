import torch


def update_H(H, L_X, epsilon=1e-8, eta=0.01):
    """
    Auxiliary-variable update from Eq. (15).

    Paper multiplicative update:
        H_ij <- H_ij * (2 L_X H)_ij /
                ((4 H H^T H)_ij + epsilon)

    eta:
        Damping factor used to stabilize the alternating optimization.
    """

    numerator = 2.0 * (L_X @ H)

    denominator = (
        4.0 * H @ H.T @ H
    ) + epsilon

    H_candidate = H * (numerator / denominator)

    # Numerical safety
    H_candidate = torch.nan_to_num(
        H_candidate,
        nan=0.0,
        posinf=10.0,
        neginf=0.0
    )

    # H must remain non-negative
    H_candidate = torch.clamp(
        H_candidate,
        min=0.0
    )

    # Damped update
    H_new = (
        (1.0 - eta) * H
        + eta * H_candidate
    )

    return H_new
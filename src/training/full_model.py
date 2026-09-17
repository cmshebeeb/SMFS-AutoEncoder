import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.graph.paper_laplacian import compute_paper_laplacian
from src.graph.auxiliary_h import update_H
from src.graph.reconstructed_structure import reconstructed_structure_loss


def differentiable_laplacian(X, sigma):
    """
    Differentiable Gaussian graph Laplacian.

    X: samples x features
    """

    # Pairwise squared Euclidean distances
    diff = X.unsqueeze(1) - X.unsqueeze(0)
    dist_sq = torch.sum(diff ** 2, dim=2)

    # Gaussian similarity
    W = torch.exp(
        -dist_sq / (2.0 * sigma ** 2)
    )

    # Remove self-similarity
    W = W - torch.diag(torch.diag(W))

    # Degree matrix
    D = torch.diag(W.sum(dim=1))

    # Graph Laplacian
    L = D - W

    return L


def estimate_sigma(X):
    """
    Estimate Gaussian bandwidth from pairwise sample distances.
    X: samples x features
    """
    diff = X.unsqueeze(1) - X.unsqueeze(0)

    dist_sq = torch.sum(diff ** 2, dim=2)

    distances = torch.sqrt(
        torch.clamp(dist_sq, min=1e-12)
    )

    nonzero = distances[distances > 0]

    return torch.median(nonzero)


def train_full_model(
    X,
    hidden_dim=128,
    alpha=0.001,
    beta=0.001,
    gamma=0.001,
    omega=0.001,
    rho1=1e-5,
    rho2=1e-5,
    num_epochs=50,
    sigma=None,
    n_neighbors=5,
    epsilon=1e-8,
    use_reconstructed_structure=True,
):

    device = X.device

    n_features = X.shape[0]
    n_samples = X.shape[1]

    # =========================================================
    # Input-data graph
    # =========================================================

    X_samples = X.T.detach().cpu().numpy()

    L_np = compute_paper_laplacian(
        X_samples,
        n_neighbors=n_neighbors,
        sigma=sigma
    )

    L_X = torch.tensor(
        L_np,
        dtype=torch.float32,
        device=device
    )

    if sigma is None:
        sigma_tensor = estimate_sigma(X.T)
        sigma_value = sigma_tensor.item()

        print(
            f"Using differentiable sigma: {sigma_value:.6f}"
        )
    else:
        sigma_value = float(sigma)

    # =========================================================
    # Initialization
    # =========================================================

    W1 = torch.randn(
        n_features,
        hidden_dim,
        device=device,
        requires_grad=True
    ) * 0.01

    W2 = torch.randn(
        hidden_dim,
        n_features,
        device=device,
        requires_grad=True
    ) * 0.01

    # Make them leaf tensors
    W1 = W1.detach().requires_grad_(True)
    W2 = W2.detach().requires_grad_(True)

    H = torch.rand(
        n_samples,
        n_samples,
        device=device
    )

    Q = torch.eye(
        n_features,
        device=device
    )

    objective_history = []
    reconstruction_history = []

    # =========================================================
    # Alternating optimization
    # =========================================================

    for epoch in range(num_epochs):

        # -----------------------------------------------------
        # A. Forward pass
        # -----------------------------------------------------

        Z = torch.sigmoid(
            W1.T @ X
        )

        X_hat = W2.T @ Z

        # -----------------------------------------------------
        # B. Reconstruction loss
        # -----------------------------------------------------

        reconstruction_loss = (
            torch.sum(
                (X - X_hat) ** 2
            ) / (2.0 * n_samples)
        )

        # -----------------------------------------------------
        # C. L2,1 surrogate
        # -----------------------------------------------------

        l21_loss = torch.trace(
            W1.T @ Q @ W1
        )

        # -----------------------------------------------------
        # D. Frobenius regularization
        # -----------------------------------------------------

        frobenius_loss = (
            torch.sum(W1 ** 2)
            + torch.sum(W2 ** 2)
        ) / 2.0

        # -----------------------------------------------------
        # E. Input graph structure
        # -----------------------------------------------------

        input_structure = torch.trace(
            W1.T
            @ X
            @ L_X
            @ X.T
            @ W1
        )

        # -----------------------------------------------------
        # F. Reconstructed graph structure
        # -----------------------------------------------------

        # Paper's reconstructed-structure term:
        #
        # Tr(W2^T Z H H^T Z^T W2)
        #
        # In our orientation:
        # X_hat = W2.T @ Z
        #
        # reconstructed_structure_loss()
        # computes the equivalent Frobenius form.

        reconstructed_structure = reconstructed_structure_loss(
            reconstructed=X_hat,
            H=H
        )

        # -----------------------------------------------------
        # G. Complete objective
        # -----------------------------------------------------

        objective = (
            reconstruction_loss
            + alpha * l21_loss
            + beta * frobenius_loss
            + (gamma / 2.0) * input_structure
            + (omega / 2.0) * reconstructed_structure
        )

        # -----------------------------------------------------
        # H. Compute W1 and W2 gradients separately
        # -----------------------------------------------------
        
        grad_W1, grad_W2 = torch.autograd.grad(
            objective,
            [W1, W2],
            retain_graph=False
        )

        print(
            f"Gradient norms: "
            f"W1={grad_W1.norm().item():.6e}, "
            f"W2={grad_W2.norm().item():.6e}"
        )

        # -----------------------------------------------------
        # I. Alternating W1/W2 update
        # -----------------------------------------------------

        with torch.no_grad():

            old_W1 = W1.clone()
            old_W2 = W2.clone()

            W1_updated = W1 - rho1 * grad_W1
            W2_updated = W2 - rho2 * grad_W2

            W1 = W1_updated.detach().clone().requires_grad_(True)
            W2 = W2_updated.detach().clone().requires_grad_(True)

            print(
                f"Relative updates: "
                f"W1={(W1-old_W1).norm().item()/(old_W1.norm().item()+1e-12):.6e}, "
                f"W2={(W2-old_W2).norm().item()/(old_W2.norm().item()+1e-12):.6e}"
            )

        # -----------------------------------------------------
        # K. Q update
        # -----------------------------------------------------

        with torch.no_grad():

            feature_norms = torch.norm(W1, dim=1)

            Q = torch.diag(
                1.0 / (
                    2.0 * feature_norms + epsilon
                )
            )

        # -----------------------------------------------------
        # L. H update
        # -----------------------------------------------------

        if epoch % 5 == 0:
            with torch.no_grad():
                H = update_H(
                    H,
                    L_X,
                    epsilon=1e-8,
                    eta=0.01
                )

                H = torch.clamp(H, min=0.0, max=10.0)
                H = H.detach()

                print(
                    f"H update: "
                    f"min={H.min().item():.6f}, "
                    f"max={H.max().item():.6f}, "
                    f"mean={H.mean().item():.6f}, "
                    f"nonzero={torch.count_nonzero(H).item()}"
                )

        # -----------------------------------------------------
        # M. Numerical safety
        # -----------------------------------------------------

        with torch.no_grad():

            W1.copy_(
                torch.nan_to_num(
                    W1,
                    nan=0.0,
                    posinf=1e3,
                    neginf=-1e3
                )
            )

            W2.copy_(
                torch.nan_to_num(
                    W2,
                    nan=0.0,
                    posinf=1e3,
                    neginf=-1e3
                )
            )

            H.copy_(
                torch.nan_to_num(
                    H,
                    nan=0.0,
                    posinf=1e3,
                    neginf=0.0
                )
            )

            H.clamp_(
                min=0.0,
                max=1e3
            )

        # -----------------------------------------------------
        # N. Recalculate objective AFTER updates
        # -----------------------------------------------------

        with torch.no_grad():

            Z_new = torch.sigmoid(
                W1.T @ X
            )

            X_hat_new = W2.T @ Z_new

            reconstruction_new = (
                torch.sum(
                    (X - X_hat_new) ** 2
                ) / (2.0 * n_samples)
            )

            l21_new = torch.trace(
                W1.T @ Q @ W1
            )

            frobenius_new = (
                torch.sum(W1 ** 2)
                + torch.sum(W2 ** 2)
            ) / 2.0

            input_structure_new = torch.trace(
                W1.T
                @ X
                @ L_X
                @ X.T
                @ W1
            )

            X_hat_new = W2.T @ Z_new

            reconstructed_structure_new = reconstructed_structure_loss(
                reconstructed=X_hat_new,
                H=H
            )

            objective_new = (
                reconstruction_new
                + alpha * l21_new
                + beta * frobenius_new
                + (gamma / 2.0)
                * input_structure_new
                + (omega / 2.0)
                * reconstructed_structure_new
            )

        objective_history.append(
            objective_new.item()
        )

        reconstruction_history.append(
            reconstruction_new.item()
        )

        print(
            f"Epoch {epoch+1}: "
            f"Recon={reconstruction_new.item():.4f}, "
            f"L21={l21_new.item():.4f}, "
            f"Frob={frobenius_new.item():.4f}, "
            f"InputGraph={input_structure_new.item():.4f}, "
            f"ReconGraph={reconstructed_structure_new.item():.4f}, "
            f"Objective={objective_new.item():.4f}"
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Reconstruction Loss: "
            f"{reconstruction_new.item():.6f}, "
            f"Objective: "
            f"{objective_new.item():.6f}"
        )

    return (
        W1,
        W2,
        H,
        Q,
        objective_history,
        reconstruction_history
    )


# =============================================================
# Colon experiment
# =============================================================

if __name__ == "__main__":

    df = pd.read_csv(
        "data/preprocessed/colon.csv"
    )

    X = df.drop(
        columns=["label"]
    ).values

    # Features × samples
    X = torch.tensor(
        X.T,
        dtype=torch.float32
    )

    # =========================================================
    # Z-score normalization
    # =========================================================

    feature_mean = X.mean(
        dim=1,
        keepdim=True
    )

    feature_std = X.std(
        dim=1,
        keepdim=True,
        unbiased=False
    )

    feature_std = torch.clamp(
        feature_std,
        min=1e-8
    )

    X = (
        X - feature_mean
    ) / feature_std

    print(
        "After Z-score normalization:"
    )

    print(
        "Mean:",
        X.mean().item()
    )

    print(
        "Std:",
        X.std(unbiased=False).item()
    )

    print(
        "Min:",
        X.min().item()
    )

    print(
        "Max:",
        X.max().item()
    )

    print(
        "Colon data shape:",
        X.shape
    )

    # =========================================================
    # Train
    # =========================================================

    (
        W1,
        W2,
        H,
        Q,
        objective_history,
        reconstruction_history
    ) = train_full_model(
        X,
        hidden_dim=128,
        alpha=0.001,
        beta=0.001,
        gamma=0.001,
        omega=0.001,
        rho1=1e-5,
        rho2=1e-5,
        num_epochs=5,
        sigma=None,
        n_neighbors=5
    )

    # =========================================================
    # Save loss curve
    # =========================================================

    figures_dir = Path(
        "reports/figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        reconstruction_history,
        label="Reconstruction Loss"
    )

    plt.plot(
        objective_history,
        label="Total Objective"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Colon - Full SMFS Model"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = (
        figures_dir
        / "colon_full_model_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================================================
    # Diagnostics
    # =========================================================

    print("\nTraining completed.")

    print(
        "Final reconstruction loss:",
        reconstruction_history[-1]
    )

    print(
        "Final objective:",
        objective_history[-1]
    )

    print(
        "W1 shape:",
        W1.shape
    )

    print(
        "W2 shape:",
        W2.shape
    )

    print(
        "H shape:",
        H.shape
    )

    print(
        "H minimum:",
        H.min().item()
    )

    print(
        "H maximum:",
        H.max().item()
    )

    print(
        "H mean:",
        H.mean().item()
    )

    print(
        "H nonzero entries:",
        torch.count_nonzero(H).item()
    )

    print(
        "Loss curve saved:",
        output_path
    )
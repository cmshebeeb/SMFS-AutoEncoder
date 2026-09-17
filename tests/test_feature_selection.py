import random

import numpy as np
import pandas as pd
import torch

from src.training.full_model import train_full_model
from src.evaluation.evaluate import evaluate_clustering


SEED = 42


def reset_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data():

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(
        "data/preprocessed/colon.csv"
    )

    y = df["label"].values

    X = torch.tensor(
        df.drop(columns=["label"]).values.T,
        dtype=torch.float32
    )

    # -----------------------------
    # Z-score normalization
    # -----------------------------
    X = (
        X - X.mean(dim=1, keepdim=True)
    ) / X.std(
        dim=1,
        keepdim=True
    ).clamp(min=1e-8)

    return X, y


def run_feature_selection(
    X,
    y,
    use_reconstructed_structure
):

    # -----------------------------
    # Train model
    # -----------------------------
    result = train_full_model(
        X,
        num_epochs=5,
        use_reconstructed_structure=use_reconstructed_structure
    )

    W1 = result[0]

    # -----------------------------
    # Feature importance
    # -----------------------------
    feature_scores = torch.norm(
        W1,
        dim=1
    )

    # -----------------------------
    # Select top 50
    # -----------------------------
    top_k = 50

    top_indices = torch.argsort(
        feature_scores,
        descending=True
    )[:top_k]

    X_selected = X[
        top_indices,
        :
    ]

    # -----------------------------
    # samples × features
    # -----------------------------
    X_cluster = X_selected.T.numpy()

    # -----------------------------
    # Clustering
    # -----------------------------
    acc, nmi = evaluate_clustering(
        X_cluster,
        y
    )

    return (
        acc,
        nmi,
        feature_scores,
        top_indices,
        X_cluster
    )


def test_feature_selection():

    # =========================================================
    # Prepare data once
    # =========================================================

    X, y = prepare_data()

    # =========================================================
    # OFF experiment
    # =========================================================

    # Reset seed so OFF starts with exactly the same
    # initialization as the ON experiment.
    reset_seed(SEED)

    (
        acc_off,
        nmi_off,
        feature_scores_off,
        top_indices_off,
        X_cluster_off
    ) = run_feature_selection(
        X,
        y,
        use_reconstructed_structure=False
    )

    # =========================================================
    # ON experiment
    # =========================================================

    # Reset the same seed again.
    # This ensures W1, W2, and H start identically.
    reset_seed(SEED)

    (
        acc_on,
        nmi_on,
        feature_scores_on,
        top_indices_on,
        X_cluster_on
    ) = run_feature_selection(
        X,
        y,
        use_reconstructed_structure=True
    )

    # =========================================================
    # Feature-selection diagnostics
    # =========================================================

    top_k = 50

    print()
    print("========================================")
    print("Reconstructed Structure Ablation")
    print("========================================")

    print()
    print("OFF:")
    print(
        f"Top-{top_k} Feature Selection Result:"
    )
    print(
        f"ACC = {acc_off:.6f}",
        flush=True
    )
    print(
        f"NMI = {nmi_off:.6f}",
        flush=True
    )

    print()
    print("ON:")
    print(
        f"Top-{top_k} Feature Selection Result:"
    )
    print(
        f"ACC = {acc_on:.6f}",
        flush=True
    )
    print(
        f"NMI = {nmi_on:.6f}",
        flush=True
    )

    # =========================================================
    # Comparison
    # =========================================================

    print()
    print("========================================")
    print("Ablation Comparison")
    print("========================================")

    print(
        f"ACC OFF = {acc_off:.6f}"
    )

    print(
        f"ACC ON  = {acc_on:.6f}"
    )

    print(
        f"ACC Δ   = {acc_on - acc_off:+.6f}"
    )

    print()

    print(
        f"NMI OFF = {nmi_off:.6f}"
    )

    print(
        f"NMI ON  = {nmi_on:.6f}"
    )

    print(
        f"NMI Δ   = {nmi_on - nmi_off:+.6f}"
    )

    print("========================================")

    # =========================================================
    # Feature overlap
    # =========================================================

    overlap = len(
        set(top_indices_off.tolist())
        &
        set(top_indices_on.tolist())
    )

    print()
    print("========================================")
    print("Feature Overlap")
    print("========================================")
    print(
        f"Common features between OFF and ON: "
        f"{overlap}/{top_k}"
    )
    print("========================================")

    # =========================================================
    # Shape checks
    # =========================================================

    assert X.shape == (
        2000,
        62
    )

    assert X_cluster_off.shape == (
        62,
        50
    )

    assert X_cluster_on.shape == (
        62,
        50
    )

    assert feature_scores_off.shape == (
        2000,
    )

    assert feature_scores_on.shape == (
        2000,
    )

    assert top_indices_off.shape == (
        50,
    )

    assert top_indices_on.shape == (
        50,
    )

    # =========================================================
    # Metric checks
    # =========================================================

    assert 0 <= acc_off <= 1
    assert 0 <= nmi_off <= 1

    assert 0 <= acc_on <= 1
    assert 0 <= nmi_on <= 1
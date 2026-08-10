import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.evaluation.evaluate import evaluate_clustering

# Load dataset
df = pd.read_csv("data/preprocessed/colon.csv")

# Separate features and labels
X = df.drop(columns=["label"]).values
y = df["label"].values

# Evaluate
acc, nmi = evaluate_clustering(X, y)

print(f"ACC: {acc:.4f}")
print(f"NMI: {nmi:.4f}")
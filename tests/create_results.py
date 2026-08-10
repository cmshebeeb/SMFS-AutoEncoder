import pandas as pd
from pathlib import Path

# Create results folder if it doesn't exist
Path("reports/results").mkdir(parents=True, exist_ok=True)

# Accuracy table
accuracy_df = pd.DataFrame({
    "Dataset": ["Colon"],
    "Baseline": [0.5161]
})

# NMI table
nmi_df = pd.DataFrame({
    "Dataset": ["Colon"],
    "Baseline": [0.0000]
})

# Save CSVs
accuracy_df.to_csv("reports/results/accuracy_results.csv", index=False)
nmi_df.to_csv("reports/results/nmi_results.csv", index=False)

print("Result files created successfully!")
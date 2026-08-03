from scipy.io import loadmat
import pandas as pd
from pathlib import Path

#Load mat file
mat_file = Path("data/raw/colon.mat")
data = loadmat(mat_file)

#print(data.keys())

#Store features and labels into variables X and Y
X = data["X"]
Y = data["Y"]
'''
print("X shape:", X.shape)
print("Y shape:", Y.shape)
'''
#create header of csv files as list
feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
#print(feature_names)


#Crete DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["label"] = Y

#save the csv file
output_file = Path("data/preprocessed/colon.csv")
df.to_csv(output_file, index=False)

#
print(df.head())
print(df.shape)
print(df["label"].unique())
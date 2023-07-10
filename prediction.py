import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv>")
    sys.exit(1)

infilename = sys.argv[1]

df = pd.read_csv(infilename, index_col="property_id")

print("Making new features...")

## Your code here
Y = df.values[:, -1]
# Getting the Lot Size
lot_size = df["lot_depth"] * df["lot_width"]
df["lot_size"] = lot_size
# Setting condition for miles_to_school, less than 2miles changes to 1 and greater than 2 changes to 0
# Applying the condition
df["is_close_to_school"] = np.where(df["miles_to_school"] < 2, 1, 0)
Df_XnewFeatures = df[["sqft_hvac", "lot_size", "is_close_to_school"]]
# Renaming the sqft_hvac to sqft
Df_XnewFeatures.rename(columns={"sqft_hvac": "sqft"}, inplace=True)
labels_XNewFeatures = Df_XnewFeatures.columns

# Finding the R2
X = np.concatenate(
    [
        Df_XnewFeatures["sqft"].values.reshape(-1, 1),
        Df_XnewFeatures["lot_size"].values.reshape(-1, 1),
        Df_XnewFeatures["is_close_to_school"].values.reshape(-1, 1),
    ],
    axis=1,
)
reg = LinearRegression().fit(X, Y)
print(f"Using only the useful ones: {[f for f in labels_XNewFeatures]}...")
print(f"R2 = {round(reg.score(X,Y),5)}")
print("*** Prediction ***")
ones = np.ones([len(Df_XnewFeatures), 1])
X = np.append(ones, Df_XnewFeatures.to_numpy(), axis=1)
B = np.linalg.inv(X.T @ X) @ X.T @ Y
pred_string = f"${round(B[0],2):,} + ({labels_XNewFeatures[0]} x ${round(B[1],2):,}) + ({labels_XNewFeatures[1]} x ${round(B[2],2):,})"
print("Price = " + pred_string)
print(
    f"\t Less than 2 miles from a school? You get ${round(B[3],2):,} added to the price!"
)

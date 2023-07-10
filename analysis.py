import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

# Deal with command-line
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv>")
    sys.exit(1)
infilename = sys.argv[1]

# Read in the basic data frame
df = pd.read_csv(infilename, index_col="property_id")
X_basic = df.values[:, :-1]
labels_basic = df.columns[:-1]
Y = df.values[:, -1]

# Expand to a 2-degree polynomials
## Your code here
poly = PolynomialFeatures(degree=2)
X_basic_T = poly.fit_transform(X_basic)

# Finding the polynomialFeatures for the X_basic_label
labels_basic_T = poly.get_feature_names(labels_basic)

# Make a new dataframe to include all the transformed PolynomialFeatures.
df_XBasic_T = pd.DataFrame(X_basic_T, columns=labels_basic_T)
# Drop the column with '1s'
df_XBasic_T = df_XBasic_T.drop(df_XBasic_T.columns[[0]], axis=1)

# Prepare for loop
residual = Y

# We always need the column of zeros to
# include the intercept
feature_indices = [0]

## Your code here for the loop
dictcolumn_pvalue = {}  # A dictionary for all Pvalue gotten to allow it sorted
lst_lowest = (
    []
)  # Create a list to have string so as to pass it to df_XBasic_T[''] to find X for the new residual
while len(feature_indices) < 3:
    # header_message ='First time through: using original price data as the residual'
    for a in range(0, len(df_XBasic_T.columns)):
        # Get Column Name
        column_name = df_XBasic_T.columns[a]

        pvalue_1 = pearsonr(df_XBasic_T.iloc[:, a], residual)

        dictcolumn_pvalue[column_name] = pvalue_1[1]

    sorted_features = {
        k: v for k, v in sorted(dictcolumn_pvalue.items(), key=lambda item: item[1])
    }

    # Getting the feature with lowest p-value after been sorted
    lowest_P_feat = list(sorted_features.keys())[
        0
    ]  # Element with the least Pvalue in the sorted list
    lst_lowest.append(lowest_P_feat)
    if len(lst_lowest) == 1:
        print("First time through: using original price data as the residual")
        for k, v in sorted_features.items():
            print("\t" + f'"{k}"' + " vs residual: p-value=" + str(v))
        print('**** Fitting with ["1"' + " " + '"' + lst_lowest[0] + '"' + "] ****")
        X = df_XBasic_T[lst_lowest[0]].values.reshape(-1, 1)
        reg = LinearRegression().fit(X, Y)
        y_hat = reg.predict(X)
        residual = Y - y_hat
        print(f"R2 = {reg.score(X,Y)}")
    elif len(lst_lowest) == 2:
        for k, v in sorted_features.items():
            print("\t" + f'"{k}"' + " vs residual: p-value=" + str(v))
        print(
            '**** Fitting with ["1"'
            + " "
            + '"'
            + lst_lowest[0]
            + '"'
            + " "
            + '"'
            + lst_lowest[1]
            + '"'
            + "] ****"
        )
        X = np.concatenate(
            [
                df_XBasic_T[lst_lowest[0]].values.reshape(-1, 1),
                df_XBasic_T[lst_lowest[1]].values.reshape(-1, 1),
            ],
            axis=1,
        )
        reg = LinearRegression().fit(X, Y)
        y_hat = reg.predict(X)
        residual = Y - y_hat
        print(f"R2 = {reg.score(X,Y)}")
    print("Residual is Updated")

    feature_indices.append(0)

# Any relationship between the final residual and the unused variables?
print("Making scatter plot: age_of_roof vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:, 3], residual, marker="+")
ax.set_title("age_of_roof vs final residual")
ax.set_xlabel("Age of Roof")
ax.set_ylabel("Final Residual")
fig.savefig("ResidualRoof.png")

print("Making a scatter plot: miles_from_school vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:, 4], residual, marker="+")
ax.set_title("miles_from_school vs final residual")
ax.set_xlabel("Miles From School")
ax.set_ylabel("Final Residual")
fig.savefig("ResidualMiles.png")

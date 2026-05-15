import pandas as pd
import os
##
print("cwd", os.getcwd())
for fn in ["Rj_Risk_Matrisi.csv", "Dij_Mesafe_Matrisi.csv"]:
    df = pd.read_csv(fn)
    print("---", fn, "---")
    print("shape", df.shape)
    print(df.head(10))
    print("unique counts", {col: df[col].nunique() for col in df.columns})

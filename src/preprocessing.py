
import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "income"
]

df = pd.read_csv(
    "data/adult.data",
    header=None,
    names=columns,
    skipinitialspace=True
)

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

#Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check unknown values represented by '?'
print("\nUnknown values represented by '?':")
print((df == '?').sum())
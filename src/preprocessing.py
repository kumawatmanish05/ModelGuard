
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
#print("\nUnknown values represented by '?':")
#print((df == '?').sum())

#Replace '?' with proper missing value
df.replace("?", pd.NA, inplace=True)

# Fill missing categorical values with the most frequent category 
categorical_columns = [
    "workclass", "occupation", "native_country"
]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Verify that missing values are handled
print(df[categorical_columns].isnull().sum())

# Check target class distribution 
print("\nIncome distribution:")
print(df["income"].value_counts())

#Check Percentages
print("\nIncome distribution percentages:")
print((df["income"].value_counts(normalize=True) * 100).round(2))


# Separate input features and target
X = df.drop(columns=["income"])
y = df["income"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget classes:")
print(y.unique())


from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))


# Identify categorical and numerical columns
categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Categorical columns:", categorical_columns)
print("\nNumerical columns:", numerical_columns)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# Define transformations for each feature type
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

print("\nPreprocessing pipeline created successfully!")


# Learn transformations from training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to test data
X_test_processed = preprocessor.transform(X_test)

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)

from sklearn.impute import SimpleImputer

# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

df.replace("?", pd.NA, inplace=True)




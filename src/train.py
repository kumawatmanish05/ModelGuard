import pandas as pd 
import joblib

from src.evaluate import evaluate_model

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)


#Load dataset
columns = [
    "age","workclass","fnlwgt","education",
    "education_num","marital_status","occupation",
    "relationship","race","sex","capital_gain",
    "capital_loss","hours_per_week","native_country","income"   
]

df = pd.read_csv(
    "data/adult.data",
    header=None,
    names=columns,
    skipinitialspace=True
)

#Mark unown values as missing
df.replace("?", float("nan"), inplace=True)

X = df.drop(columns=["income"])
y = df["income"].str.strip()  # Remove leading/trailing whitespace from target variable

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, 
    random_state=42, stratify=y
)

#Identify column types
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

#Build preprocessing pipelines for numerical and categorical features
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

#Combine preprocessing and model into a single pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000,class_weight="balanced"))
])

#Train the model
model_pipeline.fit(X_train, y_train)

print("Model training completed.")
print("Training rows:",len(X_train))
print("Testing rows:",len(X_test))

#Make predictions on the Unseen test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
metrics = evaluate_model(y_test, y_pred)

# ModelGuard quality gate
RECALL_THRESHOLD = 0.80
income_recall = metrics["recall_over_50k"]

if income_recall >= RECALL_THRESHOLD:
    print("\nModelGuard Quality Gate: PASS")
    print("Model meets the minimum recall requirement.")
else:
    raise ValueError(
        f"ModelGuard Quality Gate: FAIL. "
        f"Recall is {income_recall:.4f}, "
        f"but the minimum required is {RECALL_THRESHOLD:.2f}."
    )

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Save the trained model to a file
from pathlib import Path

# Create the 'models' directory if it doesn't exist
Path("models").mkdir(exist_ok=True)

# Save the model pipeline to a file
joblib.dump(model_pipeline, "models/modelguard_model.pkl")

print("\nTrained model saved Successfully")


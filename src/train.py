import pandas as pd 
import joblib
import mlflow

from src.evaluate import evaluate_model , RECALL_THRESHOLD

import json 
from pathlib import Path 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)

mlflow.set_experiment("ModelGuard-Income-Classification")

MODEL_PATH =Path("models/modelguard.pkl")
METRICS_PATH = Path("reports/metrics.json")


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

random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

accepted_model = None
accepted_metrics = None
accepted_model_name = None

with mlflow.start_run(run_name="LogisticRegression"):
    #Train model
    model_pipeline.fit(X_train, y_train)

    print("Model training completed.")
    print("Training rows: ",len(X_train))
    print("Testing rows: ",len(X_test))

    #Make predictions on the test set
    y_pred = model_pipeline.predict(X_test)

    #Evaluate model performance
    metrics = evaluate_model(y_test, y_pred)

    #Log model and metrics to MLflow
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metrics({"max_iter": 1000, "recall_threshold": RECALL_THRESHOLD})

    #Log Evaluation metrics
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("f1_weighted", metrics["f1_weighted"])
    mlflow.log_metric("recall_over_50k", metrics["recall_over_50k"])

    passed = metrics["quality_gate_passed"]

    if passed:
        accepted_model = model_pipeline
        accepted_metrics = metrics
        print("\nLogistic Regression accepted based on the quality gate criteria.")

    mlflow.set_tag("quality_gate","PASS" if  passed else "FAIL")
    mlflow.set_tag("model_status","candidate" if passed else "rejected")

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print(
        "\nLogistic Regression:",
        "passed the quality gate." if passed
        else "rejected by the quality gate."
    )


# Experiment 2: Random Forest
with mlflow.start_run(run_name="RandomForest"):
    random_forest_pipeline.fit(X_train, y_train)

    rf_predictions = random_forest_pipeline.predict(X_test)

    rf_metrics = evaluate_model(y_test, rf_predictions)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("recall_threshold", RECALL_THRESHOLD)

    mlflow.log_metric("accuracy", rf_metrics["accuracy"])
    mlflow.log_metric("f1_weighted", rf_metrics["f1_weighted"])
    mlflow.log_metric("recall_over_50k", rf_metrics["recall_over_50k"])

    rf_passed = rf_metrics["quality_gate_passed"]

    mlflow.set_tag("quality_gate","PASS" if rf_passed else "FAIL")
    mlflow.set_tag("model_status","candidate" if rf_passed else "rejected")

    if rf_passed:
        if (accepted_metrics is None or rf_metrics["recall_over_50k"] > accepted_metrics["recall_over_50k"]):
            accepted_model = random_forest_pipeline
            accepted_metrics = rf_metrics
            print("\nRandom Forest accepted based on the quality gate criteria.")

    else :
        print("\nRandom Forest rejected by the quality gate criteria.")

    print("\nRandom forest experiment completed.")


# Save the best model based on recall score

if accepted_model is not None:
    MODEL_PATH.parent.mkdir(parents=True ,exist_ok = True)
    
    joblib.dump(accepted_model, MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents =True ,exist_ok = True)

    final_metrics = {
        "selected_model": accepted_model_name,
        **accepted_metrics,
    }

    with open(METRICS_PATH, "w",encoding="utf-8") as file:
        json.dump(final_metrics, file, indent = 4)

    print(f"\n Selected model : {accepted_model_name}")
    print(f"Model saved to : {MODEL_PATH}")
    print(f"Metrics saved to : {METRICS_PATH}")

else:
    print("\n No Model passed the quality gate .")
    print("No new model was saved ")

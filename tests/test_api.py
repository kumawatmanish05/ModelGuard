from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")

    assert response.status_code ==200
    assert response.json() == {"status":"healthy"}


def test_predict_income():
    payload ={
        "age": 37,
        "workclass": "Private",
        "fnlwgt": 178000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married_civ_spouse",
        "occupation": "Exec_managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United_States"
    }

    response = client.post("/predict", json = payload)

    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K",">50k"]


def test_quality_check():
    response = client.get("/quality-check")

    assert response.status_code == 200

    result = response.json()

    assert result["selected_model"] == "Logistic_Regression"
    assert result["quality_gate_passed"] is True
    assert result["recall_over_50k"] >= result["required_recall"]
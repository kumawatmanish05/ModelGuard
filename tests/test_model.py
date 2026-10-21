import joblib
import pandas as pd

def test_model_loads():
    model = joblib.load("models/modelguard_model.pkl")
    assert model is not None

def test_prediction_returns_valid_class():
    model = joblib.load("models/modelguard_model.pkl")
    
    sample = {
        "age": 37,
        "workclass": "Private",
        "fnlwgt": 178000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "married_civ_spouse",
        "occupation": "Exec_managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United_States"
    }

    input_data = pd.DataFrame([sample])
    prediction = model.predict(input_data)
    
    assert prediction[0] in [">50K", "<=50K"]
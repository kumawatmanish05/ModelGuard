import joblib
import pandas as pd

#Load the saved model
model = joblib.load("models/modelguard_model.pkl")

#Example person : 14 input features
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

#Convert the sample to a DataFrame
input_data = pd.DataFrame([sample])

#Make prediction
prediction = model.predict(input_data)
print("Prediction:", prediction[0])
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_welcome():
    req = client.get('/')
    assert req.status_code == 200, "Status code is not 200"
    assert req.json() == "Welcome, this API returns predictions on Salary", "Wrong json output"


def test_get_prediction_negative():
    input_dict = {
        "age": 49,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200, "Status code is not 200"
    assert response.json() == {"Predicted salary": "<=50K"}, \
        "Wrong json output"


def test_get_prediction_positive():
    input_dict = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 1020,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200, "Status code is not 200"
    assert response.json() == {"Predicted salary": ">50K"}, "Wrong json output"

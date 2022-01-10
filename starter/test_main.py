from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_welcome():
    req = client.get('/')
    assert req.status_code == 200, "Status code is not 200"
    assert req.json() == "Welcome, this API returns predictions on Salary", "Wrong json output"


def test_get_prediction_negative():
    req = client.post(
        "/predict", json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }
    )
    assert req.status_code == 200, "Status code is not 200"
    assert req.json() == {"Predicted salary": "<=50K"}, "Wrong json output"


def test_get_prediction_positive():
    req = client.post(
        "/predict", json={
            'age': 50,
            'workclass': 'Private',
            'fnlgt': 367260,
            'education': 'Bachelors',
            'education-num': 13,
            'marital-status': 'Never-married',
            'occupation': 'Tech-support',
            'relationship': 'Unmarried',
            'race': 'White',
            'sex': 'Male',
            'capital-gain': 14084,
            'capital-loss': 0,
            'hours-per-week': 45,
            'native-country': 'Canada'
        }
    )
    assert req.status_code == 200, "Status code is not 200"
    assert req.json() == {"Predicted salary": ">50K"}, "Wrong json output"

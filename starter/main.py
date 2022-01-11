import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from starter.train_model import CAT_FEATURES
from starter.ml.model import inference
from starter.ml.data import process_data


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()


class InferenceRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get('/')
async def welcome():
    return "Welcome, this API returns predictions on Salary"


@app.post("/items")
async def create_item(item):
    return item


@app.post('/predict')
async def get_prediction(request_data: InferenceRequest = Body(
    ...,
    example={
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
)):
    cwd_p = os.getcwd()
    trained_model = joblib.load(f"{cwd_p}/starter/model/model_trained.joblib")
    encoder = joblib.load(f"{cwd_p}/starter/model/encoder.joblib")
    labels = joblib.load(f"{cwd_p}/starter/model/lb.joblib")

    request_dict = request_data.dict(by_alias=True)
    request_df = pd.DataFrame(request_dict, index=[0])
    processed_data, _, _, _ = process_data(
        request_df, categorical_features=CAT_FEATURES, label=None,
        training=False, encoder=encoder, lb=labels
    )
    preds = inference(trained_model, np.array(processed_data))
    return {"Predicted salary": preds[0]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)

import math
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .ml import model


@pytest.fixture
def generate_dummy_data():
    data_df = pd.DataFrame({
        "id": [1, 2, 3],
        "numerical_feat": [3.0, 15.0, 25.9],
        "target_feat": ["yes", "no", "no"]
    })
    return data_df


def test_train_model(data):
    X = data[['numerical_feat']]
    y = data['target_feat']
    trained_model = model.train_model(X, y)
    assert isinstance(
        trained_model, RandomForestClassifier), "Wrong model type"


def test_compute_model_metrics():
    y = [1, 0, 0]
    y_preds = [1, 1, 0]
    precision, recall, fbeta = model.compute_model_metrics(y, y_preds)
    assert precision == 0.5, "Precision incorrect"
    assert recall == 1.0, "Recall incorrect"
    assert math.isclose(fbeta, 0.6666, rel_tol=1e-04), "fbeta incorrect"


def test_inference(data):
    X = data[['numerical_feat']]
    y = data['target_feat']
    fake_model = RandomForestClassifier()
    fake_model_ = fake_model.fit(X, y)
    preds = model.inference(fake_model_, X)
    assert isinstance(preds, np.ndarray), "The output type is not np.ndarray"
    assert len(preds) == X.shape[0], "Length of the output is not matched"

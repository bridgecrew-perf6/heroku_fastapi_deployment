import math
from random import random
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .ml import model


@pytest.fixture
def dummy_data():
    data_df = pd.DataFrame({
        "id": list(range(100)),
        "numerical_feat": [random()*100 for i in range(100)],
    })
    data_df["target_feat"] = [
        "yes" if i > 80 else "no" for i in data_df["numerical_feat"].values]
    return data_df


def test_train_model(dummy_data):
    X = dummy_data[['numerical_feat']]
    y = dummy_data['target_feat']
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


def test_inference(dummy_data):
    X = dummy_data[['numerical_feat']]
    y = dummy_data['target_feat']
    dummy_model = RandomForestClassifier()
    _dummy_model = dummy_model.fit(X, y)
    preds = model.inference(_dummy_model, X)
    assert isinstance(preds, np.ndarray), "The output type is not np.ndarray"
    assert len(preds) == X.shape[0], "Length of the output is not matched"

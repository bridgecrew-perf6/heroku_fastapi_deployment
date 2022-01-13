import pandas as pd
from joblib import load
from ml import model, data
from train_model import CAT_FEATURES


def compute_score(datapath='../data/census_cleaned.csv'):
    data_df = pd.read_csv(datapath)

    model_trained = load("../model/model_trained.joblib")
    encoder = load("../model/encoder.joblib")
    label = load("../model/lb.joblib")

    X_test, y_test, _, _ = data.process_data(
        data_df,
        categorical_features=CAT_FEATURES,
        label="salary", encoder=encoder, lb=label, training=False)
    y_preds = model_trained.predict(X_test)
    prc, rcl, fb1 = model.compute_model_metrics(y_test, y_preds)

    metrics_df = pd.DataFrame(
        {"precision": prc, "recall": rcl, "fbeta_score": fb1}, index=[0])

    metrics_df.to_csv('../model/model_metrics.txt', index=False)


if __name__ == '__main__':
    compute_score()

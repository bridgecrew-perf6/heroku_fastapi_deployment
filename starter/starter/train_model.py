# Script to train machine learning model.
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load
from ml import model, data

# Add the necessary imports for the starter code.
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def train(datapath='data/census_cleaned.csv'):
    # Add code to load in the data.
    data_df = pd.read_csv(datapath)
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train_df, _ = train_test_split(data_df, test_size=0.20)

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, label = data.process_data(
        train_df, categorical_features=CAT_FEATURES,
        label="salary", training=True
    )

    # Train and save a model.
    model_trained = model.train_model(X_train, y_train)

    dump(model_trained, "../model/model_trained.joblib")
    dump(encoder, "../model/encoder.joblib")
    dump(label, "../model/lb.joblib")


def compute_score_sliced(datapath='data/census_cleaned.csv'):
    data_df = pd.read_csv(datapath)
    _, test = train_test_split(data_df, test_size=0.20)

    model_trained = load("../model/model_trained.joblib")
    encoder = load("../model/encoder.joblib")
    label = load("../model/lb.joblib")

    sliced_scores_df = pd.DataFrame(
        columns=["feature", "value", "precision", "recall", "fbeta_score"])
    for cat in CAT_FEATURES:
        for val in test[cat].unique():
            df_temp = test[test[cat] == val]

            X_test, y_test, _, _ = data.process_data(
                df_temp,
                categorical_features=CAT_FEATURES,
                label="salary", encoder=encoder, lb=label, training=False)

            y_preds = model_trained.predict(X_test)

            prc, rcl, fb1 = model.compute_model_metrics(y_test, y_preds)

            line = \
                f"[{cat} to {val}] Precision: {prc} Recall: {rcl} FBeta: {fb1}"
            logging.info(line)
            sliced_scores_df = sliced_scores_df.append({
                    "feature": cat, "value": val, "precision": prc,
                    "recall": rcl, "fbeta_score": fb1
                }, ignore_index=True
            )
    sliced_scores_df.to_csv('../model/slice_output.txt', index=False)

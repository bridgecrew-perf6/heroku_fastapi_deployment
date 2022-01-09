# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from .ml import model, data

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
    train, test = train_test_split(data_df, test_size=0.20)

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, label = data.process_data(
        train, categorical_features=CAT_FEATURES,
        label="is_salary_over50k", training=True
    )

    # Train and save a model.
    model_trained = model.train_model(X_train, y_train)

    dump(model_trained, "../model/gb_model.joblib")
    dump(encoder, "../model/encoder.joblib")
    dump(label, "../model/lb.joblib")

import pathlib
import numpy as np
import pandas as pd


DATA_PATH = pathlib.Path('../data/census.csv')


def clean_data():
    data_df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    data_df.replace({'?': np.nan}, inplace=True)
    data_df.dropna(inplace=True)
    return data_df


if __name__ == "__main__":
    df = clean_data()
    df.to_csv(DATA_PATH.parent/'census_cleaned.csv', index=False)

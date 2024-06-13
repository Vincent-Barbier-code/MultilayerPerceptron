import pandas as pd
import numpy as np

from data_processing import set_random as set_random


def data_true(data: pd.DataFrame) -> pd.DataFrame:
    """Validation df function"""
    data.drop(data.columns[range(2, data.shape[1])], axis=1, inplace=True)
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.replace({"M": "0", "B": "1"})
    return data


def data_feature(data: pd.DataFrame) -> pd.DataFrame:
    """Train df function"""
    data.drop(data.columns[[0, 1]], axis=1, inplace=True)
    # normalize the train_data
    return (data - data.mean()) / data.std()

def create_dfs(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the dataframes"""

    train_data, validation_data, test_data = set_random.set_random_data(dataframe)
    
    train_data.to_csv("../data/mydata/train_data.csv", header=False, index=False)
    validation_data.to_csv("../data/mydata/validation_data.csv", header=False, index=False)
    test_data.to_csv("../data/mydata/test_data.csv", header=False, index=False)

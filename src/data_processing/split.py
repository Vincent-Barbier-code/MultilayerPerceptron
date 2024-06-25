import pandas as pd

from data_processing import set_random as set_random
from data_processing.set_random import sort_data

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

    train_data, validation_data = set_random.set_random_data(dataframe)
    
    train_data.to_csv("../data/mydata/train_data.csv", header=False, index=False)
    validation_data.to_csv("../data/mydata/validation_data.csv", header=False, index=False)

def create_test_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the test data"""

    dfB, dfM = sort_data(df)
    frac = 0.8

    dfB_train = dfB.sample(frac=frac, random_state=41)
    dfM_train = dfM.sample(frac=frac, random_state=41)

    dfB_test = dfB.drop(dfB_train.index)
    dfM_test = dfM.drop(dfM_train.index)

    test_data = pd.concat([dfB_test, dfM_test])
    test_data = test_data.sample(frac=1, random_state=41).reset_index(drop=True)

    test_true = data_true(test_data.copy())
    test_data = data_feature(test_data)

    df = pd.concat([dfB_train, dfM_train])

    return test_data, test_true, df
"""Module for setting random training and validation data"""

import pandas as pd


def shuffle_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Shuffle the data"""

    return dataframe.sample(frac=1).reset_index(drop=True)

def sort_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort the data"""

    dataframe.sort_values(by=dataframe.columns[1], inplace=True)
    dfB = dataframe.loc[dataframe[dataframe.columns[1]] == "B"]
    dfM = dataframe.loc[dataframe[dataframe.columns[1]] == "M"]
    return dfB, dfM

def set_random_data(
        dataframe: pd.DataFrame, frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    dfB, dfM = sort_data(dataframe)

    dfB_train = dfB.sample(frac=frac)
    dfM_train = dfM.sample(frac=frac)

    dfB_validation = dfB.drop(dfB_train.index)
    dfM_validation = dfM.drop(dfM_train.index)

    train_data = pd.concat([dfB_train, dfM_train])
    validation_data = pd.concat([dfB_validation, dfM_validation])

    train_data = shuffle_data(train_data)
    validation_data = shuffle_data(validation_data)

    return train_data, validation_data

"""Module for setting random training and validation data"""

import pandas as pd


def shuffle_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Shuffle the data"""

    return dataframe.sample(frac=1, random_state=424).reset_index(drop=True)

def sort_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort the data"""

    dataframe.sort_values(by=dataframe.columns[1], inplace=True)
    dfB = dataframe.loc[dataframe[dataframe.columns[1]] == "B"]
    dfM = dataframe.loc[dataframe[dataframe.columns[1]] == "M"]
    return dfB, dfM

def set_random_data(
        dataframe: pd.DataFrame, frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    dfB, dfM = sort_data(dataframe)

    dfB_train = dfB.sample(frac=frac)
    dfM_train = dfM.sample(frac=frac)

    dfB_validation = dfB.drop(dfB_train.index)
    dfM_validation = dfM.drop(dfM_train.index)

    dfB_test = dfB_train.sample(frac=0.3)
    dfM_test = dfM_train.sample(frac=0.3)

    dfB_train = dfB_train.drop(dfB_test.index)
    dfM_train = dfM_train.drop(dfM_test.index)

    train_data = pd.concat([dfB_train, dfM_train])
    validation_data = pd.concat([dfB_validation, dfM_validation])
    test_data = pd.concat([dfB_test, dfM_test])

    train_data = shuffle_data(train_data)
    validation_data = shuffle_data(validation_data)
    test_data = shuffle_data(test_data)

    return train_data, validation_data, test_data

# def set_random_data(
#     dataframe: pd.DataFrame, frac: float = 0.8
# ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """Set random training and validation data"""
    
#     if frac > 1 or frac < 0:
#         raise ValueError("Fraction must be between 0 and 1")
#     dataframe = dataframe.sample(frac=1).reset_index(drop=True)
#     train_data = dataframe.sample(frac=frac)
#     validation_data = dataframe.drop(train_data.index)
#     test_data = train_data.sample(frac=0.1)
#     train_data = train_data.drop(test_data.index)
#     return train_data, validation_data, test_data

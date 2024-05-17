"""Module for setting random training and validation data"""

import numpy as np
import pandas as pd

def shuffle_data(dataframe: pd.DataFrame) -> pd.DataFrame:
	"""Shuffle the data"""

	return dataframe.sample(frac=1).reset_index(drop=True)

def set_random_data(dataframe: pd.DataFrame, frac: float=0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Set random training and validation data"""

	if frac > 1 or frac < 0:
		raise ValueError("Fraction must be between 0 and 1")
	dataframe = dataframe.sample(frac=1).reset_index(drop=True)
	train_data = dataframe.sample(frac=frac)
	validation_data = dataframe.drop(train_data.index)
	return train_data, validation_data
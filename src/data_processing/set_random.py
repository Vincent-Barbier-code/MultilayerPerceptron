"""Module for setting random training and validation data"""

import numpy as np
import pandas as pd


def set_random_data(dataframe: pd.DataFrame) -> tuple:
	"""Set random training and validation data"""

	dataframe = dataframe.sample(frac=1).reset_index(drop=True)
	train_data = dataframe.sample(frac=0.6)
	validation_data = dataframe.drop(train_data.index)
	return train_data, validation_data
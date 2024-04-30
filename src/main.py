"""Main module"""

from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute
from data_processing.set_random import set_random_data
from data_processing import data_extractor as extractor
from perceptron.neural import Neural

import numpy as np

def main() -> None:
	"""Main function"""

	args = arg_parsing()
	execute(args)
	dataframe = extractor.extract_data("data/data.csv")

	df_features = dataframe.drop(columns=[0, 1])


	neural = Neural(df_features.values)
	neural.add_layer(4, "sigmoid")
	neural.add_layer(5, "sigmoid")
	neural.add_layer(6, "relu")

	for i in range(len(neural.layers)):
		print(neural.layers[i].W)

if __name__ == "__main__":
	main()

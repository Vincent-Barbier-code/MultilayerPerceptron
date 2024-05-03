"""Main module"""

from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute
from data_processing.set_random import set_random_data
from data_processing import extract
from perceptron.neural import Neural

import numpy as np


def main() -> None:
    """Main function"""

    args = arg_parsing()
    execute(args)
    dataframe = extract.Extract("data/data.csv", header=None).data

    df_features = dataframe.copy()
    df_features.drop(dataframe.columns[[0, 1]], axis=1, inplace=True)

    neural = Neural(df_features.values)
    neural.add_layer(4, 0)
    neural.add_layer(4, 0, "sigmoid")
    # neural.add_layer(5, "sigmoid")
    # neural.add_layer(6, "relu")

    for i in range(len(neural.layers)):
        print(neural.layers[i])


if __name__ == "__main__":
    main()

"""Main module"""

from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute
from data_processing import extract
from perceptron.neural import Neural

import numpy as np

def main() -> None:
    """Main function"""

    args = arg_parsing()
    execute(args)

    np.random.seed(42)

    dataframe = extract.Extract(args.file, header=None).data

    df_features = dataframe.copy()
    df_features.drop(dataframe.columns[[0, 1]], axis=1, inplace=True)

    # normalize the data
    df_features = (df_features - df_features.mean()) / df_features.std()

    dataframe.drop(
        dataframe.columns[range(2, dataframe.shape[1])], axis=1, inplace=True
    )
    dataframe.drop(dataframe.columns[0], axis=1, inplace=True)
    dataframe = dataframe.replace({"M": "0", "B": "1"})

    neural = Neural(df_features.values, epoch=100, learning_rate=0.01, batch_size=50)
    neural.add_layer(10, "relu")
    neural.add_layer(20, "relu")
    neural.add_layer(20, "relu")
    neural.add_layer(20, "relu")
    neural.add_layer(20, "sigmoid")
    neural.add_layer(1, "softmax")
    neural.train(df_features.values, dataframe.values)


if __name__ == "__main__":
    main()

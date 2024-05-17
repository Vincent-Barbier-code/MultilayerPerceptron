"""Main module"""

from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute
from data_processing import extract
from perceptron.neural import Neural
from data_processing import set_random

import pandas as pd
import numpy as np

def  data_true(data: pd.DataFrame) -> pd.DataFrame:
    """Validation function"""
    data.drop(data.columns[range(2, data.shape[1])], axis=1, inplace=True)
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.replace({"M": "0", "B": "1"})
    return data

def data_feature(data: pd.DataFrame) -> pd.DataFrame:
    """Train function"""
    data.drop(data.columns[[0,1]], axis=1, inplace=True)
    # normalize the train_data
    return (data - data.mean()) / data.std()

def main() -> None:
    """Main function"""

    args = arg_parsing()
    execute(args)

    np.random.seed(42)

    dataframe = extract.Extract(args.file, header=None).data
    dataframe = set_random.shuffle_data(dataframe)

    train_data, validation_data = set_random.set_random_data(dataframe)

    # Train data
    train_true = data_true(train_data.copy())
    train_data = data_feature(train_data.copy())

    # Validation data
    validation_true = data_true(validation_data.copy())
    validation_data = data_feature(validation_data.copy())

    # Neural network
    neural = Neural(train_data.values, epoch=10000, learning_rate=0.0001, batch_size=64)
    neural.add_layer(24, "sigmoid")
    neural.add_layer(24, "sigmoid")
    neural.add_layer(24, "sigmoid")
    neural.add_layer(1, "softmax")
    neural.train(train_data.values, train_true.values)
    
    neural.predict(validation_data.values, validation_true.values)

if __name__ == "__main__":
    main()

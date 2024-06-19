"""Main module"""

from data_processing import extract
from data_processing.split import data_true, data_feature
from perceptron.network import Network
from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute

import numpy as np
import pandas as pd
import pickle


def train(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Train the network network

    Args:
        train_data (pd.DataFrame): The train data.
        test_data (pd.DataFrame): The test data.

    Returns:
        None: None.
    """

    # Train data
    train_true = data_true(train_data.copy())
    train_data = data_feature(train_data)

    # Test data
    test_true = data_true(test_data.copy())
    test_data = data_feature(test_data)

    network = Network(epoch=1000, learning_rate=0.001, batch_size=93, optimizer="Adam")
    network.add_layer(40, "relu")
    
    network.add_layer(5, "sigmoid")
    network.add_layer(2, "softmax")
    network.train(train_data.values, train_true.values, test_data.values, test_true.values)

    # np.random.seed(42)
    # network = Network(epoch=1000, learning_rate=0.001, batch_size=93, optimizer="None")
    # network.add_layer(40, "relu")
    
    # network.add_layer(5, "sigmoid")
    # network.add_layer(2, "softmax")
    # network.train(train_data.values, train_true.values, test_data.values, test_true.values)

    pickle.dump(network, open("../data/mymodels/network.pkl", "wb"))
    print("> saving model '../data/mymodels/network.pkl' to disk...")


def predict(validation_data: pd.DataFrame) -> None:
    """Predict the validation data"""

    # Validation data
    validation_true = data_true(validation_data.copy())
    validation_data = data_feature(validation_data)

    network = pickle.load(open("../data/mymodels/network.pkl", "rb"))
    network.predict(validation_data.values, validation_true.values)


def main() -> None:
    """Main function"""

    np.random.seed(42)    
    # print(np.random.get_state())
    args = arg_parsing()
    execute(args)

    # Train data
    if args.train:
        train_data = extract.Extract(args.file).data
        test_data = extract.Extract(args.file2).data
        train(train_data, test_data)

    # Predict
    if args.predict:
        validation_data = extract.Extract(args.file).data
        predict(validation_data)


if __name__ == "__main__":
    main()

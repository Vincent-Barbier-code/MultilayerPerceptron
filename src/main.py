"""Main module"""

import numpy as np
import pandas as pd
import pickle


from data_processing import extract
from data_processing.split import data_true, data_feature
from perceptron.network import Network
from perceptron.benchmark import benchmark
from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute
from data_visualization.plots import loss_acc_plot
from data_visualization.plots import metrics_plot
from data_processing.split import create_test_data
from test.sklearn_test import train_sklearn
from test.sklearn_test import predict_sklearn


def train(train_data: pd.DataFrame) -> None:
    """Train the network network

    Args:
        train_data (pd.DataFrame): The train data.

    Returns:
        None: None.
    """

    # Test data
    test_data, test_true, train_data = create_test_data(train_data)

    # Train data
    train_true = data_true(train_data.copy())
    train_data = data_feature(train_data)

    network = Network(epoch=1000, learning_rate=0.01, batch_size=16)
    network.add_layer(24, "sigmoid")
    network.add_layer(24, "sigmoid")
    network.add_layer(24, "sigmoid")
    network.add_layer(2, "softmax")
    network.train(
        train_data.values, train_true.values, test_data.values, test_true.values
    )

    loss_acc_plot(network)
    if arg_parsing().metrics:
        metrics_plot(network)

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

    np.random.seed(41)

    args = arg_parsing()
    execute(args)

    actions = {
        "train": train,
        "predict": predict,
        "benchmark": benchmark,
        "sklearn": train_sklearn,
        "predict_sklearn": predict_sklearn,
    }

    train_data = args.file
    validation_data = args.file2

    # Train data
    if args.train:
        actions["train"](train_data)

    # Predict
    elif args.predict:
        actions["predict"](validation_data)

    # Benchmark
    elif args.benchmark:
        actions["benchmark"](train_data, validation_data)

    # Sklearn
    elif args.sklearn:
        actions["sklearn"](train_data)
        actions["predict_sklearn"](validation_data)

    # Train and predict
    elif args.all:
        actions["train"](train_data)
        actions["predict"](validation_data)


if __name__ == "__main__":
    main()

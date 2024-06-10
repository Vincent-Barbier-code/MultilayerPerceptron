"""Main module"""

from data_processing import extract
from data_processing.split import data_true, data_feature
from perceptron.neural import Neural
from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute

import numpy as np
import pandas as pd
import pickle


def train(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Train the neural network"""

    # Train data
    train_true = data_true(train_data.copy())
    train_data = data_feature(train_data)

    # Test data
    test_true = data_true(test_data.copy())
    test_data = data_feature(test_data)

    neural = Neural(train_data.values, epoch=50, learning_rate=0.01, batch_size=20)
    neural.add_layer(18, "relu")
    neural.add_layer(13, "sigmoid")
    neural.add_layer(8, "sigmoid")
    neural.add_layer(2, "softmax")
    neural.train(train_data.values, train_true.values, test_data.values, test_true.values)

    pickle.dump(neural, open("../data/mymodels/neural.pkl", "wb"))
    print("> saving model '../data/mymodels/neural.pkl' to disk...")


def predict(validation_data: pd.DataFrame) -> None:
    """Predict the validation data"""

    # Validation data
    validation_true = data_true(validation_data.copy())
    validation_data = data_feature(validation_data)

    neural = pickle.load(open("../data/mymodels/neural.pkl", "rb"))
    neural.predict(validation_data.values, validation_true.values)


def main() -> None:
    """Main function"""

    np.random.seed(42)

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

import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

from data_processing.split import data_true, data_feature


import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

from data_processing.split import data_true, data_feature


def train_sklearn(train_data: pd.DataFrame) -> None:
    train_true = data_true(train_data.copy())
    train_data = data_feature(train_data)

    mlp = MLPClassifier(
        hidden_layer_sizes=(24, 24, 24),
        activation="relu",
        solver="sgd",
        learning_rate="constant",
        max_iter=1000,
        learning_rate_init=0.01,
        batch_size=16,
        early_stopping=True,
        validation_fraction=0.2,
        shuffle=True,
    )
    mlp.fit(train_data.to_numpy(), train_true.to_numpy().ravel())

    with open("../data/mymodels/sklearn_network.pkl", "wb") as file:
        pickle.dump(mlp, file)
    print("> saving model '../data/mymodels/sklearn_network.pkl' to disk...")


def predict_sklearn(validation_data: pd.DataFrame) -> None:
    validation_true = data_true(validation_data.copy())
    validation_features = data_feature(validation_data)

    with open("../data/mymodels/sklearn_network.pkl", "rb") as file:
        mlp = pickle.load(file)
    mlppred = mlp.predict_proba(validation_features.to_numpy())
    print(
        f"Accuracy: {mlp.score(validation_features.to_numpy(), validation_true.to_numpy().ravel()):.2f}"
    )
    log_loss_value = log_loss(validation_true, mlppred)
    print(f"Log loss: {log_loss_value:.2f}")


def predict_sklearn(validation_data: pd.DataFrame) -> None:
    """Predict the validation data using the trained model

    Args:
        validation_data (pd.DataFrame): The validation data.

    Returns:
        None: None.
    """
    # Validation data
    validation_true = data_true(validation_data.copy())
    validation_data = data_feature(validation_data)

    # Load the trained model
    mlp = pickle.load(open("../data/mymodels/sklearn_network.pkl", "rb"))
    mlppred = mlp.predict_proba(validation_data.values)
    print(f"Accuracy: {mlp.score(validation_data.values, validation_true):.2f}")
    log_loss_value = log_loss(validation_true, mlppred)
    print(f"Log loss: {log_loss_value:.2f}")

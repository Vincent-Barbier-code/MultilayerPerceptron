import numpy as np
import pandas as pd

from perceptron.layer import Layer


class Neural:

    def __init__(
        self,
        X: np.ndarray,
        alpha: float = 0.1,
        epoch: int = 100,
        batch_size: int = 10,
        learning_rate: float = 0.1,
    ) -> None:
        self.X = X
        self.alpha = alpha
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = []
        self.layers = []  # type: list[Layer]
        self.accuracy = []

    def add_layer(
        self,
        Next: int = 0,
        f_activation: str = "sigmoid",
        weights_init: str = "heUniform",
    ) -> None:
        if len(self.layers) == 0:
            layer = Layer(Next, 0, f_activation, weights_init)
        else:
            layer = Layer(Next, self.layers[-1].Next, f_activation, weights_init)
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation through the neural network.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The output of the neural network."""

        for layer in self.layers:
            X = layer.forward(X)

        return X

    def cost(self, input: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Computes the cost function for the neural network.

        Args:
            P (np.ndarray): The predicted labels.
            Y (np.ndarray): The true labels.

        Returns:
            float: The cost of the neural network."""
        epsilon = 1e-9  # to avoid log(0)
        input = np.clip(input, epsilon, 1 - epsilon)
        Y = Y.astype(float)
        Z = (
            -1
            / len(Y)
            * np.sum(
                Y * np.log(input + epsilon) + (1 - Y) * np.log(1 - input + epsilon),
                axis=0,
            )
        )
        return Z

    def backward(self, P: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Backward propagation through the neural network.

        Args:
            P (np.ndarray): The predicted labels.
            Y (np.ndarray): The true labels."""

        cost = self.cost(P, Y)
        for layer in reversed(self.layers):
            cost = layer.backward(layer.input, cost, self.learning_rate)

        return cost

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train the neural network."""

        P = self.forward(X)
        self.backward(P, Y)

import numpy as np

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
        self.layers = []
        self.accuracy = []

    def add_layer(
        self,
        N: int = 0,
        M: int = 0,
        f_activation: str = "sigmoid",
        weights_init: str = "heNormal",
    ) -> None:
        if N == 0:
            raise ValueError("Number of neurons must be greater than 0")
        layer = Layer(N, M, f_activation, weights_init)
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:

        return

    def loss_function(self, X: np.ndarray, y: np.ndarray) -> float:
        return 0.1

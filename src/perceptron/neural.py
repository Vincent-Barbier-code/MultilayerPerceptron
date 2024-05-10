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

    def cost(self, Y: np.ndarray, P: np.ndarray) -> float:
        """Computes the cost function for the neural network.

        Args:
            Y (np.ndarray): The true labels.
            P (np.ndarray): The predicted labels.

        Returns:
            float: The cost of the neural network."""

        print("True label = " + str(Y) + ";\n" + "Predicted label = " + str(P))
        Z = float(
            -1 / self.layers[-1].Next * np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
        )
        print(Z)
        return Z

    # def backward(self, X: np.ndarray, Y: np.ndarray) -> None:
    #     """Backward propagation through the neural network.

    #     Args:
    #         X (np.ndarray): The input data.
    #         Y (np.ndarray): The true labels."""

    #     P = self.forward(X)
    #     dA = P - Y

    #     for layer in reversed(self.layers):
    #         dA = layer.backward(dA, self.learning_rate)

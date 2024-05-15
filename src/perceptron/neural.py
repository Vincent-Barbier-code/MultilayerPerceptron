import numpy as np

from perceptron.layer import Layer


class Neural:

    def __init__(
        self,
        X: np.ndarray,
        alpha: float = 0.1,
        epoch: int = 30,
        batch_size: int = 256,
        learning_rate: float = 0.1,
    ) -> None:
        self.X = X
        self.alpha = alpha
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.layers: list[Layer] = []

    def add_layer(
        self,
        Next: int = 0,
        f_activation: str = "sigmoid",
        weights_init: str = "zeros",
    ) -> None:
        if len(self.layers) == 0:
            layer = Layer(Next, 0, f_activation, weights_init)
        else:
            layer = Layer(Next, self.layers[-1].Next, f_activation, weights_init)
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation through the neural network.

        Args:
            X (np.ndarray): The Z data.

        Returns:
            np.ndarray: The output of the neural network."""

        for layer in self.layers:
            X = layer.forward(X)
        return X

    def loss(self, P: np.ndarray, Y: np.ndarray) -> float:
        """Computes the loss function for the neural network.

        Args:
            P (np.ndarray): The predicted labels.
            Y (np.ndarray): The true labels.

        Returns:
            float: The loss of the neural network."""
        epsilon = 1e-9  # to avoid log(0)

        loss = (
            -1
            / len(Y)
            * np.sum(
                Y * np.log(P + epsilon) + (1 - Y) * np.log(1 - P + epsilon),
                axis=0,
                keepdims=True,
            )
        )
        return loss

    def backward(self, P: np.ndarray, Y: np.ndarray) -> None:
        """Backward propagation through the neural network.

        Args:
            P (np.ndarray): The predicted labels.
            Y (np.ndarray): The true labels."""

        loss = self.loss(P, Y)
        gradient = P - Y
        for layer in reversed(self.layers):
            gradient = layer.backward(loss, self.learning_rate, gradient)
        
    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute the accuracy of the neural network.

        Args:
            X (np.ndarray): The Z data.
            Y (np.ndarray): The true labels.

        Returns:
            float: The accuracy of the neural network."""
        P = self.forward(X)
        P = np.round(P)
        accuracy = np.sum(P == Y) / len(Y)

        return accuracy


    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train the neural network."""

        # Convert Y["0", "1", ...] into float
        Y = Y.astype(float)

        for i in range(self.epoch):
            for j in range(0, len(X), self.batch_size):
                self.X = X[j : j + self.batch_size]
                self.Y = Y[j : j + self.batch_size]
                P = self.forward(self.X)
                self.backward(P, self.Y)
        
            accuracy = self.accuracy(self.X, self.Y)
            print(f"Accuracy: {accuracy}")      

import numpy as np

from perceptron.layer import Layer

def one_hot(a: np.ndarray, num_classes: int):
    """Convert an array of integers into a one-hot encoded matrix."""
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).astype(int)

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
            X (np.ndarray): The Z data.

        Returns:
            np.ndarray: The output of the neural network."""

        for layer in self.layers:
            X = layer.forward(X)
        return X

    def loss(self, P: np.ndarray, Y: np.ndarray) -> None:
        """Computes the loss function for the neural network.

        Args:
            P (np.ndarray): The predicted labels.
            Y (np.ndarray): The true labels.

        Returns:
            float: The loss of the neural network."""
            
        epsilon = 1e-9  # to avoid log(0)

        loss = - np.mean(
            # -1
            # / len(P)
             np.sum(
                Y * np.log(P + epsilon) + (1 - Y) * np.log(1 - P + epsilon),
                axis=1,
            )
        )
        print(f"Loss : {loss}")
        
    def BCE(self, y_true, y_pred, eps: float = 1e-16):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        log_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(log_loss)

    def backward(self, P: np.ndarray, Y: np.ndarray) -> None:
        """Backward propagation through the neural network.

        Args:
            P (np.ndarray): The predicted labels.
            Y (np.ndarray): The true labels."""

        Yreshape = one_hot(Y, 2)
        gradient = P - Yreshape
        
        for layer in reversed(self.layers):
            gradient = layer.backward(self.learning_rate, gradient)

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Compute the accuracy of the neural network.

        Args:
            X (np.ndarray): The Z data.
            Y (np.ndarray): The true labels.

        Returns:
            float: The accuracy of the neural network."""
        
        P = self.forward(X)
        P = np.argmax(P, axis=1)
        Y = Y.reshape(-1)
        accuracy = sum(P == Y) / len(Y)
        print(f"Accuracy : {accuracy}")

    def shuffle_batch(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shuffle the data and return a batch of data.

        Args:
            X (np.ndarray): The X data.
            Y (np.ndarray): The true labels.

        Returns:
            tuple[np.ndarray, np.ndarray]: The shuffled data and labels."""
        

        idx = np.random.permutation(len(X))
        return X[idx], Y[idx]

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train the neural network."""

        # Convert Y["0", "1", ...] into float
        Y = Y.astype(int)
        for _ in range(self.epoch):
            X, Y = self.shuffle_batch(X, Y)
            for j in range(0, len(X), self.batch_size):
                
                self.X = X[j : j + self.batch_size]
                self.Y = Y[j : j + self.batch_size]
            
                P = self.forward(self.X)
                self.backward(P, self.Y)

                Y_one = one_hot(self.Y, 2)
                print(f"Loss1 : {self.BCE(Y_one, P)}")
                self.loss(P, Y_one)
            
        self.accuracy(X, Y)

    def predict(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Predict the labels of the data."""

        Y = Y.astype(int)

        self.Y = one_hot(Y, 2)
        self.loss(self.forward(X), self.Y)
        self.accuracy(X, Y)

        
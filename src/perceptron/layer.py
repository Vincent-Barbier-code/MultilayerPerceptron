import numpy as np


def sigmoid(X: np.ndarray) -> np.ndarray:
    """Applies the sigmoid activation function to an input array.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the sigmoid function.
    """

    return 1 / (1 + np.exp(-X))


def softmax(X: np.ndarray) -> np.ndarray:
    """Applies the softmax function to an input array.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the softmax function.
    """

    X = np.exp(X)
    S = np.sum(X)
    return X / S


def relu(X: np.ndarray) -> np.ndarray:
    """Applies the ReLU (Rectified Linear Unit) function to an input array.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the ReLU function.
    """

    return np.maximum(0, X)


def weight_initialization(M: int, N: int, weights_init: str) -> np.ndarray:
    """Initializes weights based on the specified method.

    Args:
        M (int): The number of neurons in the previous layer.
        N (int): The number of neurons in the current layer.
        weights_init (str): The method to use for weight initialization. Options are "random", "zeros", "heUniform", and "heNormal".

    Returns:
        np.ndarray: The initialized weights.

    Raises:
        ValueError: If an invalid weight initialization method is provided.
    """

    match weights_init:
        case "random":
            W = np.random.rand(M, N)
        case "zeros":
            W = np.zeros((M, N))
        case "heUniform":
            W = np.random.randn(M, N) * np.sqrt(2 / N)
        case "heNormal":
            W = np.random.randn(M, N) / np.sqrt(N)
        case _:
            raise ValueError(f"Invalid activation function: {weights_init}")

    return W


class Layer:
    """A class to represent a layer in a neural network."""

    def __init__(
        self,
        M: int,
        N: int,
        f_activation: str = "sigmoid",
        weights_init: str = "heNormal",
    ) -> None:
        """Initializes a Layer with the specified parameters.

        Args:
            N (int, optional): The number of neurons in the layer.
            f_activation (str, optional): The activation function to use. Defaults to "sigmoid".
            weights_init (str, optional): The method to use for weight initialization. Defaults to "heNormal".
        """

        self.N = N
        self.W = weight_initialization(M, N, weights_init)
        self.bias = np.zeros(N)
        self.f_activation = f_activation

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Performs the forward pass for the layer.

        Args:
            X (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array after applying the activation function.
        """

        Z = np.dot(X, self.W) + self.bias

        match self.f_activation:
            case "sigmoid":
                return sigmoid(Z)
            case "softmax":
                return softmax(Z)
            case "relu":
                return relu(Z)
            case _:
                raise ValueError(f"Invalid activation function: {self.f_activation}")

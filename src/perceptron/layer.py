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


def weight_initialization(Next: int, Previous: int, weights_init: str) -> np.ndarray:
    """Initializes weights based on the specified method.

    Args:
        Previous (int): The number of neurons in the previous layer.
        Next (int): The number of neurons in the current layer.
        weights_init (str): The method to use for weight initialization. Options are "random", "zeros", "heUniform", and "heNormal".

    Returns:
        np.ndarray: The initialized weights.

    Raises:
        ValueError: If an invalid weight initialization method is provided.
    """

    if Previous == 0:
        return np.zeros((Previous, Next))

    match weights_init:
        case "random":
            W = np.random.rand(Previous, Next)
        case "zeros":
            W = np.zeros((Previous, Next))
        case "heUniform":
            W = np.random.randn(Previous, Next) * np.sqrt(2 / Next)
        case "heNormal":
            W = np.random.randn(Previous, Next) / np.sqrt(Next)
        case _:
            raise ValueError(f"Invalid activation function: {weights_init}")

    return W


class Layer:
    """A class to represent a layer in a neural network."""

    def __init__(
        self,
        Next: int = 0,
        Previous: int = 0,
        f_activation: str = "sigmoid",
        weights_init: str = "heUniform",
    ) -> None:
        """Initializes a Layer with the specified parameters.

        Args:
            Next (int, optional): The number of neurons in the layer.
            Previous (int, optional): The number of neurons in the previous layer.
            f_activation (str, optional): The activation function to use. Defaults to "sigmoid".
            weights_init (str, optional): The method to use for weight initialization. Defaults to "heNormal".
        """

        if Next == 0:
            raise ValueError("Number of neurons in entry must be greater than 0")
        self.Next = Next
        self.Previous = Previous
        self.W = weight_initialization(Next, Previous, weights_init)
        self.bias = np.zeros(Next)
        self.f_activation = f_activation

    def forward(self, X: np.ndarray) -> None:
        """Performs the forward pass for the layer.

        Args:
            X (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array after applying the activation function.
        """
        if self.Previous == 0:
            self.Previous = X.shape[1]
            self.W = weight_initialization(self.Next, self.Previous, "heUniform")

        # Z is the output layer before
        self.Z = X

        self.A = np.dot(X, self.W) + self.bias
        match self.f_activation:
            case "sigmoid":
                X = sigmoid(self.A)
            case "softmax":
                X = softmax(self.A)
            case "relu":
                X = relu(self.A)
            case _:
                raise ValueError(f"Invalid activation function: {self.f_activation}")

        self.A = X

    def backward(self, loss: float, learning_rate: float, gradient: np.ndarray) -> np.ndarray:
        """Performs the backward pass for the layer.

        Args:
            derivateGradient (np.ndarray): The gradient of the loss function.
            learning_rate (float): The learning rate.

        Returns:
            np.ndarray: The gradient of the loss function.
        """

        



        return gradient

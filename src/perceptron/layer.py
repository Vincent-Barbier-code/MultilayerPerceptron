import numpy as np

from perceptron.optimizer import Optimizer


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

    EX = np.exp(X - np.max(X, axis=1, keepdims=True))
    return EX / EX.sum(axis=1, keepdims=True)


def relu(X: np.ndarray) -> np.ndarray:
    """Applies the ReLU (Rectified Linear Unit) function to an input array.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the ReLU function.
    """

    return np.maximum(0, X)


def tanh(X: np.ndarray) -> np.ndarray:
    """Applies the tanh function to an input array.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the tanh function.
    """

    return np.tanh(X)


def d_sigmoid(X: np.ndarray) -> np.ndarray:
    """Computes the derivative of the sigmoid function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the derivative of the sigmoid function.
    """

    return sigmoid(X) * (1 - sigmoid(X))


def d_softmax(X: np.ndarray) -> np.ndarray:
    """Computes the derivative of the softmax function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the derivative of the softmax function.
    """

    # Z = softmax(X)
    # Z * (1 - Z)
    return 1


def d_relu(X: np.ndarray) -> np.ndarray:
    """Computes the derivative of the ReLU function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the derivative of the ReLU function.
    """

    return np.where(X > 0, 1, 0)


def d_tanh(X: np.ndarray) -> np.ndarray:
    """Computes the derivative of the tanh function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the derivative of the tanh function.
    """

    return 1 - np.tanh(X) ** 2


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
    """A class to represent a layer in a network network."""

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
        self.weights_init = weights_init
        self.bias = np.zeros(Next)
        self.f_activation = f_activation

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Performs the forward pass for the layer.

        Args:
            X (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array after applying the activation function.
        """
        if self.Previous == 0:
            self.Previous = X.shape[1]
            self.W = weight_initialization(self.Next, self.Previous, self.weights_init)

        # Z is the output layer before
        self.input = X
        self.Z = np.dot(self.input, self.W) + self.bias
        match self.f_activation:
            case "sigmoid":
                X = sigmoid(self.Z)
            case "softmax":
                X = softmax(self.Z)
            case "relu":
                X = relu(self.Z)
            case "tanh":
                X = tanh(self.Z)
            case _:
                raise ValueError(f"Invalid activation function: {self.f_activation}")
        return X

    def backward(self, optimizer: Optimizer, gradient: np.ndarray) -> np.ndarray:
        """Performs the backward pass for the layer.

        Args:
            derivateGradient (np.ndarray): The gradient of the loss function.
            learning_rate (float): The learning rate.

        Returns:
            np.ndarray: The gradient of the loss function.
        """
        
        activation = None
        match self.f_activation:
            case "sigmoid":
                activation = d_sigmoid(self.Z)
            case "relu":
                activation = d_relu(self.Z)
            case "softmax":
                activation = d_softmax(self.Z)
            case "tanh":
                activation = d_tanh(self.Z)
            case _:
                raise ValueError(f"Invalid activation function: {self.f_activation}")

        
        ngradient = gradient * activation
        self.dW = np.dot(self.input.T, ngradient)
        self.dbias = np.sum(ngradient, axis=0)

        dgradient = np.dot(ngradient, self.W.T)
        optimizer.update(self)


        return dgradient

import numpy as np

def sigmoid(X: np.ndarray) -> np.ndarray:

	return (1 / (1 + np.exp(-X)))

def softmax(X: np.ndarray) -> np.ndarray:
	
	X = np.exp(X)
	S = np.sum(X)
	return (X / S)

def relu(X: np.ndarray) -> np.ndarray:
	
	return (np.maximum(0, X))


def weight_initialization(N: int, weights_init: str, seed: int=None) -> np.ndarray:
	
	np.random.seed(seed) if seed is not None else None 
	match weights_init:
		case "random":
			W = np.random.rand(N)
		case "zeros":
			W = np.zeros(N)
		case "heUniform":
			W = np.random.randn(N) * np.sqrt(2/N)
		case "heNormal":
			W = np.random.randn(N) / np.sqrt(N)
	return (W)

class Perceptron:
	
	def __init__(self, N: int, alpha: float=0.1, bias: float=0, weights_init: str="heNormal", seed: int=0) -> None:
		
		self.alpha = alpha
		self.bias = bias
		self.W = None
		self.W = weight_initialization(N, weights_init, seed)

	def activation(self, X:np.ndarray, method: str = "sigmoid") -> np.ndarray:
	
		func = None
		match method:
			case "sigmoid":
				func = sigmoid
			case "softmax":
				func = softmax
			case "relu":
				func = relu

		return func(X)

	def step(self, X: np.ndarray) -> np.ndarray:

		Wsum = np.dot(X, self.W) + self.bias
		y = self.activation(Wsum)
		return(y)
	
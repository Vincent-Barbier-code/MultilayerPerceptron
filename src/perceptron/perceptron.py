import numpy as np

def sigmoid(X: np.ndarray) -> np.ndarray:

	return (1 / (1 + np.exp(-X)))

def softmax(X: np.ndarray) -> np.ndarray:

	X = np.exp(X)
	print(X)
	return (X)

class Perceptron:
	
	def __init__(self, N: int, alpha: float=0.1) -> None:
	
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha

	def activation(self, X:np.ndarray, method: str = "sigmoid") -> None:
	
		func = None
		match method:
			case "sigmoid":
				func = sigmoid
			case "softmax":
				func = softmax

		return func(X)

	
	

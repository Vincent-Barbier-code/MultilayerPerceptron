import numpy as np

def sigmoid(X: np.ndarray) -> np.ndarray:

	return (1 / (1 + np.exp(-X)))

def softmax(X: np.ndarray) -> np.ndarray:
	
	X = np.exp(X)
	S = np.sum(X)
	return (X / S)

def relu(X: np.ndarray) -> np.ndarray:

	return (np.maximum(0, X))

def weight_initialization(N: int, weights_init: str) -> np.ndarray:
	
	match weights_init:
		case "random":
			W = np.random.rand(N)
		case "zeros":
			W = np.zeros(N)
		case "heUniform":
			W = np.random.randn(N) * np.sqrt(2/N)
		case "heNormal":
			W = np.random.randn(N) / np.sqrt(N)
		case _:
			raise ValueError(f"Invalid activation function: {weights_init}")
		
	return (W)

class Layer:

	def __init__(self, N:int=0, f_activation:str="sigmoid", weights_init:str="heNormal") -> None:
		
		self.N = N
		self.W = weight_initialization(N, weights_init)
		self.bias = np.zeros(N)
		self.f_activation = f_activation

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		outputs = self.step(inputs)
		return outputs

	def step(self, X: np.ndarray) -> np.ndarray:
		Wsum = np.dot(X, self.W.T) + self.bias
		y = self.activation(Wsum, self.f_activation)
		return(y)

	def activation(self, X:np.ndarray, method: str = "sigmoid") -> np.ndarray:
	
		func:function
		match method:
			case "sigmoid":
				func = sigmoid
			case "softmax":
				func = softmax
			case "relu":
				func = relu
			case _:
				raise ValueError(f"Invalid activation function: {method}")

		return func(X)
	

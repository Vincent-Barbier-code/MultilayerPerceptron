import copy
import numpy as np


class EarlyStop:
	"""Early stopping class."""

	def __init__(self, patience: int) -> None:
		self.cmp = 0
		self.patience = patience
		self.best_network = None

	def early_stop(self, network, val_losses: list[float]) -> bool:
		"""Early stopping function.

		Args:
			Network (Network): The network network.
			val_losses (list[float]): The validation losses.
			patience (int): The patience.

		Returns:	
			bool: If the network network should stop early.
			
		"""
		
		min_val_loss = min(val_losses)
		if val_losses[-1] > min_val_loss:
			self.cmp += 1
		if val_losses[-1] == min_val_loss:
			self.cmp = 0
			self.best_network = copy.deepcopy(network.layers)
		if self.cmp == self.patience:
			return True
		return False


class Optimizer:
	"""Class to optimize the network network.
		Default optimizer is Stochastic Gradient Descent (SGD).
	"""

	def __init__(self, learning_rate) -> None:
		self.learning_rate = learning_rate


	def update(self, layer) -> None:
		layer.W -= self.learning_rate * layer.dW
		layer.bias -= self.learning_rate * layer.dbias


def calc_prev(prev, beta, deriv):
	prev = beta * prev + (1 - beta) * deriv
	return prev

def hat(prev, beta):
	return prev / (1 - beta)

class Adam(Optimizer):
	"""Adam optimizer class"""

	def __init__(self, learning_rate: float) -> None:
		super().__init__(learning_rate)
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.epsilon = 1e-8
		self.m = []
		self.v = []
	
	def update(self, layer):
		
		self.m.append(np.zeros(layer.W.shape))
		self.v.append(np.zeros(layer.W.shape))
		
		self.m[-1] = calc_prev(self.m[-1], self.beta1, layer.dW)
		self.v[-1] = calc_prev(self.v[-1], self.beta2, layer.dW**2)

		m_hat = hat(self.m[-1], self.beta1)
		v_hat = hat(self.v[-1], self.beta2)
		layer.W -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

		self.m.append(np.zeros(layer.bias.shape))
		self.v.append(np.zeros(layer.bias.shape))

		self.m[-1] = calc_prev(self.m[-1], self.beta1, layer.dbias)
		self.v[-1] = calc_prev(self.v[-1], self.beta2, layer.dbias**2)
		
		m_hat = hat(self.m[-1], self.beta1)
		v_hat = hat(self.v[-1], self.beta2)
		layer.bias -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class Rmsprop(Optimizer):
	"""Rmsprop optimizer class"""

	def __init__(self, learning_rate: float) -> None:
		super().__init__(learning_rate)
		self.beta = 0.9
		self.epsilon = 1e-8
		self.dW = []
		self.dbias = []

	def update(self, layer):
		self.dW.append(np.zeros(layer.W.shape))
		self.dbias.append(np.zeros(layer.bias.shape))

		self.dW[-1] = calc_prev(self.dW[-1], self.beta, layer.dW ** 2)
		self.dbias[-1] = calc_prev(self.dbias[-1], self.beta, layer.dbias ** 2)

		layer.W -= self.learning_rate * layer.dW / (np.sqrt(self.dW[-1]) + self.epsilon)
		layer.bias -= self.learning_rate * layer.dbias / (np.sqrt(self.dbias[-1]) + self.epsilon)

	
class Momentum(Optimizer):
	"""Momentum optimizer class"""

	def __init__(self, learning_rate: float) -> None:
		super().__init__(learning_rate)
		self.beta = 0.9
		self.m = []

	def update(self, layer):
		"""Update the weights and biases of the layer.

		Args:
			layer (Layer): The layer.
		"""
		self.m.append(np.zeros(layer.W.shape))
		self.m[-1] = calc_prev(self.m[-1], self.beta, layer.dW)

		m_hat = hat(self.m[-1], self.beta)
		layer.W -= self.learning_rate * m_hat		
		self.m.append(np.zeros(layer.bias.shape))
		self.m[-1] = calc_prev(self.m[-1], self.beta, layer.dbias)
		layer.W -= self.learning_rate * m_hat
		
   
def create_optimizer(name: str, learning_rate: float) -> Optimizer:
	"""Create an optimizer.

	Args:
		optimizer (str): The optimizer.
		learning_rate (float): The learning rate.

	Returns:
		Optimizer: The optimizer.
	"""

	optimizer = {
		"SGD": Optimizer,
		"Momentum": Momentum,
		"Adam": Adam,
		"Rmsprop": Rmsprop,
	}
	if name not in optimizer:
		raise ValueError(f"Invalid optimizer: {name}")
	else:
		return optimizer[name](learning_rate)
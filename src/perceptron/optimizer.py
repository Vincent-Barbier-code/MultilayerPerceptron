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
	"""Class to optimize the network network."""

	def __init__(self, learning_rate) -> None:
		self.learning_rate = learning_rate


	def update(self, W, bias, dW: np.ndarray, dbias: np.ndarray) -> None:
     
		W -= self.learning_rate * dW
		bias -= self.learning_rate * dbias

		return W, bias
	
# class Adam(Optimizer):


# class Rmsprop(Optimizer):


class Momentum(Optimizer):
	"""Momentum optimizer class"""

	def __init__(self, learning_rate: float) -> None:
		super().__init__(learning_rate)
		self.Beta = 0.9
		self.v = []

	def update(self, W, bias, dW, dbias) -> None:
		"""Update the network network.

		Args:
			network (Network): The network network.
		"""
  
		if not self.v:
			self.v = [np.zeros_like(dW), np.zeros_like(bias)]
		self.v[0] = self.Beta * self.v[0] + (1 - self.Beta) * dW
		self.v[1] = self.Beta * self.v[1] + (1 - self.Beta) * bias
		W -= self.learning_rate * self.v[0]
		bias -= self.learning_rate * self.v[1]

		return W, bias
		
   
def create_optimizer(name: str | None, learning_rate: float) -> Optimizer:
	"""Create an optimizer.

	Args:
		optimizer (str): The optimizer.
		learning_rate (float): The learning rate.

	Returns:
		Optimizer: The optimizer.
	"""
	opt = [None, "momentum"]
	optimizer = {
		None: Optimizer,
		"momentum": Momentum,
	}
	if name not in opt:
		raise ValueError(f"Invalid optimizer: {name}")
	else:
		
		return optimizer[name](learning_rate)
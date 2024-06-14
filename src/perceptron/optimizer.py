import copy
from perceptron.network import Network
class EarlyStop:
	"""Early stopping class."""

	def __init__(self, patience: int) -> None:
		self.cmp = 0
		self.patience = patience
		self.best_network = None

	def early_stop(self, network: Network, val_losses: list[float]) -> bool:
		"""Early stopping function.

		Args:
			Network (Network): The network network.
			val_losses (list[float]): The validation losses.
			patience (int): The patience.

		Returns:
			None: If the network network should not stop early.
			Network: If the network network should stop early. The network network to use.
		"""
		
		min_val_loss = min(val_losses)
		if val_losses[-1] > min_val_loss:
			self.cmp += 1
		if val_losses[-1] == min_val_loss:
			self.cmp = 0
			self.best_network = copy.deepcopy(network)
		if self.cmp == self.patience:
			return True
		return False


class Optimizer:
	"""Class to optimize the network network."""

	def __init__(self, Beta, v, learning_rate) -> None:
		self.best_network = None
		self.cmp = 0
		self.Beta = Beta
		self.v = v
		self.learning_rate = learning_rate



	
# class Adam(Optimizer):


# class Rmsprop(Optimizer):


class Momentum(Optimizer):
	"""Momentum optimizer class"""

	def __init__(self, Beta, learning_rate) -> None:
		super().__init__(Beta, 0, learning_rate)

	def update(self, network: Network) -> None:
		"""Update the network network.

		Args:
			network (Network): The network network.
		"""
		for i, layer in enumerate(network.layers):
			if i == 0:
				continue
			layer.weights += self.learning_rate * layer.dweights + self.Beta * self.v
			layer.biases += self.learning_rate * layer.dbiases + self.Beta * self.v
		self.v = self.learning_rate * layer.dweights + self.Beta * self.v
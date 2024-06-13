import copy
from perceptron.network import Network

class Optimizer:
	"""Class to optimize the network network."""

	def __init__(self) -> None:
		self.best_network = None
		self.cmp = 0
	def early_stop(self, network: Network, val_losses: list[float], patience: int) -> Network | None:
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
		if val_losses[-1] >= min_val_loss:
			self.cmp += 1
		if val_losses[-1] == min_val_loss:
			self.cmp = 0
			self.best_network = copy.deepcopy(network)
		if self.cmp == patience:
			return self.best_network
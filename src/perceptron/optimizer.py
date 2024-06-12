from perceptron.neural import Neural

class Optimizer:
	"""Class to optimize the neural network."""

	def __init__(self) -> None:
		self.neurals = []

	def early_stop(self, neural: Neural, val_losses: list[float], patience: int) -> Neural | None:
		"""Early stopping function.

		Args:
			neural (Neural): The neural network.
			val_losses (list[float]): The validation losses.
			patience (int): The patience.

		Returns:
			bool: True if the model should stop, False otherwise."""
		
		epoch = len(val_losses)
		cmp = 0
		if epoch < patience:
			self.neurals.append(neural)
			return None
		for i in range(0, epoch - patience):
			cmp = 0
			for j in range(i, i + patience):
				if val_losses[i] < val_losses[j]:
					cmp += 1
					if cmp == patience:
						return self.neurals[i]
		self.neurals.append(neural)
		return None
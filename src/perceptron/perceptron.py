"""Module for the Perceptron class"""
import numpy as np

class Perceptron:
	def __init__(self, N: int, alpha: float=0.1) -> None:
		"""Initialize the perceptron with N weights and alpha learning rate"""
		self.weight = np.random.randn(N + 1) / np.sqrt(N) 
		self.alpha = alpha

	def step(self, x: int) -> int:
		"""Step function"""

		return 1 if x > 0 else 0
	
	def fit(self, X : np.ndarray, y: np.ndarray, epochs: int=10) -> None:
		"""Fit the perceptron"""
		X = np.c_[X, np.ones((X.shape[0]))]
		for epoch in np.arange(0, epochs):
			for (x, target) in zip(X, y):
				p = self.step(np.dot(x, self.weight))
				if p != target:
					error = p - target
					self.weight += -self.alpha * error * x

	def predict(self, X: np.ndarray, addBias=True) -> int:
		"""Predict the perceptron"""
		X = np.atleast_2d(X)
		if addBias:
			X = np.c_[X, np.ones((X.shape[0]))]
		return self.step(np.dot(X, self.weight))
	
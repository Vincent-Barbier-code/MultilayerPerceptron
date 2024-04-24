import numpy as np
from perceptron.perceptron import Perceptron

class Layer:

	def __init__(self, N: int, N_percep:int=24, activation:str="sigmoid") -> None:
		self.N = N
		self.N_percep = N_percep
		self.activation = activation
		self.perceptrons = []
		for i in range (N_percep):
			self.perceptrons.append(Perceptron(N))

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		outputs = np.zeros(self.N_percep)
		for i, perceptron in enumerate(self.perceptrons):
			outputs[i] = perceptron.step(inputs)
		print(outputs)
		return outputs


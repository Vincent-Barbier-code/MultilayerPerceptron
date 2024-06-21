import pandas as pd
import numpy as np
import copy

from perceptron.network import Network
from data_processing.split import create_test_data, data_true, data_feature
from data_visualization.plots import bench_plots

def create_networks() -> list[Network]:
	"""Create networks with differents optimizers"""
	network = Network(epoch=1000, learning_rate=0.001, batch_size=16)
	network.add_layer(24, "sigmoid")
	network.add_layer(24, "sigmoid")
	network.add_layer(2, "softmax")
	networks = [network, copy.deepcopy(network), copy.deepcopy(network), copy.deepcopy(network)]

	return networks

def benchmark(train_data: pd.DataFrame, validation_data: pd.DataFrame) -> None:
	"""Benchmark the networks"""

	# Test data
	test_data, test_true, train_data = create_test_data(train_data)

	# Train data
	train_true = data_true(train_data.copy())
	train_data = data_feature(train_data)

	# Validation data
	validation_true = data_true(validation_data.copy())
	validation_data = data_feature(validation_data)

	networks = create_networks()
	for network, name in zip(networks, ["SGD", "Momentum", "Adam", "Rmsprop"]):
		np.random.seed(41)
		network.train(train_data.values, train_true.values, test_data.values, test_true.values, opt=name)
		print(f"{name} done.")
		network.predict(validation_data.values, validation_true.values)
	bench_plots(networks)
	
	print("Benchmark done.")

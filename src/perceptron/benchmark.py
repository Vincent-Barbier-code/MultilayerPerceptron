import pandas as pd
import numpy as np

from perceptron.network import Network
from data_processing.split import create_test_data, data_true, data_feature
from data_visualization.plots import bench_plots

def create_networks() -> dict[str, Network]:
    """Create networks with differents optimizers"""

    networks = {"SGD": None, "Adam": None, "RMSprop": None}
    for name in networks.keys():
        np.random.seed(42)
        networks[name] = (Network(epoch=1000, learning_rate=0.001, batch_size=16, optimizer=name))
        networks[name].add_layer(24, "sigmoid")
        networks[name].add_layer(24, "sigmoid")
        networks[name].add_layer(24, "sigmoid")
        networks[name].add_layer(2, "softmax")

    return networks

def benchmark(train_data: pd.DataFrame, validation_data: pd.DataFrame) -> None:
	"""Benchmark the networks"""

	# Test data
	test_data, test_true = create_test_data(train_data)

	# Train data
	train_true = data_true(train_data.copy())
	train_data = data_feature(train_data)
 
	# Validation data
	validation_true = data_true(validation_data.copy())
	validation_data = data_feature(validation_data)

	networks = create_networks()
	for name, network in networks.items():
		network.train(train_data.values, train_true.values, test_data.values, test_true.values)
		print(f"{name} done.")
		network.predict(validation_data.values, validation_true.values)
	bench_plots(networks)
	
	print("Benchmark done.")

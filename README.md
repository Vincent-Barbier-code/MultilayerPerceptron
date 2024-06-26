# Multilayer-Perceptron
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

## Description
This is a simple implementation of a Multilayer Perceptron (MLP) using Python and Numpy. The MLP is trained using the backpropagation algorithm. 

## Objective
- To understand the working of a Multilayer Perceptron.
- To implement a Multilayer Perceptron from scratch.
- To understand the working of gradient descent.
- To understand the backpropagation algorithm.

## Installation
1. Clone the repository
2. cd into src
3. Run the following command to install the required dependencies
   python main.py --install

## Usage
1. cd into src
2. Run the following command to separate the data set into two parts, one for training and
   the other for validation
   python main.py --split or python main.py --split --file /path/to/dataset
3. Run the following command to train the model
   python main.py --train or python main.py --train --file /path/to/data_training
4. Run the following command to predict using the model
   python main.py --predict or python main.py --predict --file /path/to/data_validation
5. Run the following command to visualize the benchmark
   python main.py --benchmark --file /path/to/data_training --file2 /path/to/data_validation
6. Run the following command to train the model and predict using the model
   python main.py --all --file /path/to/data_training --file2 /path/to/data_validation
7. Run the following to test on sklearn
   python main.py --sklearn --file /path/to/data_training --file2 /path/to/data_validation

For each of the above commands, you can specify the following optional arguments:
- --file: The path to the dataset. Default is data.csv
- --early_stop: if set, the training will stop when the validation loss starts increasing

For the command train, you can specify an other optional arguments:
- --metrics: if set, the model will print the metrics for each epoch

To see a portion of the data set, you can run the following command:
   python main.py --visualize --file /path/to/dataset

## Clean up
1. Run the following command to remove the generated files
   python main.py --clean
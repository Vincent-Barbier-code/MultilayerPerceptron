PLOTS_TYPE ?= all

help:
	@echo "Available commands:"
	@echo "  help       Show this help message"	
	@echo "  install    Install the required packages"
	@echo "  all        Install, train and clean"
	@echo "  run        Run the application"
	@echo "  train      Train the neural network model"
	@echo "  test       Run the tests"
	@echo "  plots      Create plots"
	@echo "  clean      Clean up temporary files"

install:
	@echo "Installing the required packages..."
	pip install -r requirements.txt

all: install train clean

run:
	@echo "Running the program..."
	python3 src/main.py

train:
	@echo "Training the neural network model..."
	python3 src/train.py

test:
	@echo "Running the tests..."
	PYTHONPATH=./src python3 -m pytest ./src/tests

plots:
	@echo "Creating plots..."
	@mkdir -p plots
	@mkdir -p plots/scatter_plots
	@mkdir -p plots/heat_maps
	python3 src/data_visualization/plots.py $(PLOTS_TYPE)

%:
	@:

clean:
	@echo "Cleaning up temporary files..."
	@rm -rf plots

.PHONY: help install train clean plots test run all
.PHONY: help install train clean

help:
	@echo "Available commands:"
	@echo "  help       Show this help message"	
	@echo "  install    Install the required packages"
	@echo "  all        Install, train and clean"
	@echo "  run        Run the application"
	@echo "  train      Train the neural network model"
	@echo "  test       Run the tests"
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
	PYTHONPATH=./src python3 -m pytest tests

clean:
	@echo "Cleaning up temporary files..."


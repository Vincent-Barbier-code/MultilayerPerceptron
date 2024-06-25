"""This module contains the parsing of args"""

import argparse
import subprocess
import os
import inspect


def execute(args: argparse.Namespace) -> None:
    """Execute the command"""

    commands = {
        "install": install,
        "split": split,
        "train": train,
        "predict": predict,
        "clean": clean,
        "benchmark": benchmark,
        "visualize": visualize,
    }

    for arg, func in commands.items():
        if getattr(args, arg):
            if len(inspect.signature(func).parameters) == 0:
                func()  # Appelle la fonction sans arguments
            else:
                func(args)  # Appelle la fonction avec args comme argument
            break


def arg_parsing() -> argparse.Namespace:
    """Parse the arguments passed to the program"""

    parser = argparse.ArgumentParser(
        prog="Multilayer Perceptron",
        description="An artificial network networks with multiple layers \
        to predict whether a cancer is malignant or benign on a dataset of breast cancer diagnosis.",
    )

    parser.add_argument(
        "--file", type=str, help="Path to the CSV file"
    )
    parser.add_argument(
        "--file2", type=str, help="Path to the CSV file2"
    )
    parser.add_argument(
        "--install", action="store_true", help="Install the required packages"
    )
    parser.add_argument(
        "--split", action="store_true", help="Split the data into training and validation sets"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the network network model"
    )
    parser.add_argument(
        "--early_stop", action="store_true", help="Early stop the training"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the data"
    )
    parser.add_argument("--predict", action="store_true", help="Predict the validation data")
    parser.add_argument("--clean", action="store_true", help="Clean up temporary files")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the network network")
    parser.add_argument("--metrics", action="store_true", help="Print the metrics")

    return parser.parse_args()


def install() -> None:
    """Install the required packages"""
    print("Installing the required packages...")
    subprocess.run(["pip", "install", "-r", "../requirements.txt"])
    exit(0)

def split(args: argparse.Namespace) -> None:
    """Split the data into training and validation sets"""
    
    print("Splitting the data into training and validation sets...")
    create_dirs(["../data/mydata"])
    
    if args.file is None:
        args.file = "../data/data.csv"
    
    import data_processing as process
    
    dataframe = process.extract.Extract(args.file).data
    process.split.create_dfs(dataframe)
    exit(0)

def visualize(args: argparse.Namespace) -> None:
    """Visualize the data"""
    print("Visualizing the data...")

    import data_processing as process
    from data_visualization import plot_data
    
    args.file = verif_file(args.file)
    dataframe = process.extract.Extract(args.file).data
    plot_data(dataframe)

    exit(0)

def train(args: argparse.Namespace) -> None:
    """Train the network network model"""
    print("Training the network network model...")
    args.file = verif_file(args.file)
    args.early_stop = False if args.early_stop == None else True

    create_dirs(["../data/mymodels"])

def predict(args: argparse.Namespace) -> None:
    """Predict the validation data"""
    print("Predicting the validation data...")
    args.file = verif_file(args.file, "../data/mydata/validation_data.csv")

    if not os.path.exists("../data/mymodels/network.pkl"):
        print("No model found. Train the model first.")
        exit(1)

def clean() -> None:
    """Clean up temporary files"""
    print("Cleaning up temporary files...")
    subprocess.run(["rm", "-rf", "plots"])
    subprocess.run(["rm", "-rf", "../data/mydata"])
    subprocess.run(["rm", "-rf", "../data/mymodels"])
    exit(0)

def create_dirs(dir_list):
    """Helper function to create directories."""
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)

def benchmark(args: argparse.Namespace) -> None:
    """Benchmark the network network"""
    print("Benchmarking the network network...")
    args.file = verif_file(args.file)
    args.file2 = verif_file(args.file2, "../data/mydata/validation_data.csv")
    
    create_dirs(["../data/mymodels"])
    
def verif_file(file, default="../data/mydata/train_data.csv") -> str:

    if file is None:
        file = default
        if not os.path.exists(file):
            print("No file found for training. Split the data first.")
            exit(1)
    return file
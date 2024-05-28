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
        "sc": sc,
        "hm": hm,
        "pp": pp,
        "plot": plot,
        "clean": clean,
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
        description="An artificial neural networks with multiple layers \
        to predict whether a cancer is malignant or benign on a dataset of breast cancer diagnosis.",
    )

    parser.add_argument(
        "--file", type=str, help="Path to the CSV file"
    )
    parser.add_argument(
        "--install", action="store_true", help="Install the required packages"
    )
    parser.add_argument(
        "--split", action="store_true", help="Split the data into training and validation sets"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the neural network model"
    )
    parser.add_argument("--predict", action="store_true", help="Predict the validation data")
    parser.add_argument("--sc", action="store_true", help="Create scatter plots")
    parser.add_argument("--hm", action="store_true", help="Create heat maps")
    parser.add_argument("--pp", action="store_true", help="Create pair plots")
    parser.add_argument("--plot", action="store_true", help="Create all plots")
    parser.add_argument("--clean", action="store_true", help="Clean up temporary files")

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


def train(args: argparse.Namespace) -> None:
    """Train the neural network model"""
    print("Training the neural network model...")
    if args.file is None:
        args.file = "../data/mydata/train_data.csv"
        if not os.path.exists(args.file):
            print("No file found for training. Split the data first.")
            exit(1)
    create_dirs(["../data/mymodels"])
    if os.path.exists("../data/mymodels/neural.pkl"):
        os.remove("../data/mymodels/neural.pkl")
        

def predict(args: argparse.Namespace) -> None:
    """Predict the validation data"""
    print("Predicting the validation data...")
    if args.file is None:
        args.file = "../data/mydata/validation_data.csv"
        if not os.path.exists(args.file):
            print("No file found for predicting. Split the data first.")
            exit(1)
    if not os.path.exists("../data/mymodels/neural.pkl"):
        print("No model found. Train the model first.")
        exit(1)

def plot(args: argparse.Namespace) -> None:
    """Create all plots"""
    import data_visualization as dv

    print("Creating all plots...")
    if args.file:
        create_dirs(
            ["plots", "plots/scatter_plots", "plots/heat_maps", "plots/pair_plots"]
        )
        dv.plots("hm")
        dv.plots("sc")
        dv.plots("pp")
    else:
        print("No file specified for creating plots. Use default csv file.")


def hm(args: argparse.Namespace) -> None:
    """Create heat maps"""
    import data_visualization as dv

    print("Creating heat maps...")
    if args.file:
        create_dirs(["plots", "plots/heat_maps"])
        dv.plots("hm")
    else:
        print("No file specified for creating heat maps. Use default csv file.")


def sc(args: argparse.Namespace) -> None:
    """Create scatter plots"""
    import data_visualization as dv

    print("Creating scatter plots...")
    if args.file:
        create_dirs(["plots", "plots/scatter_plots"])
        dv.plots("sc")
    else:
        print("No file specified for creating scatter plots. Use default csv file.")


def pp(args: argparse.Namespace) -> None:
    """Create pair plots"""
    import data_visualization as dv

    print("Creating pair plots...")
    if args.file:
        create_dirs(["plots", "plots/pair_plots"])
        dv.plots("pp")
    else:
        print("No file specified for creating pair plots. Use default csv file.")


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

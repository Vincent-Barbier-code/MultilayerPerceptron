"""This module contains the parsing of args"""

import argparse
import subprocess
import os
import inspect


def execute(args: argparse.Namespace)-> None:
    """Execute the command"""

    commands = {
        'install': install,
        'run': run, # Not really implemented
        'train': train, # Not implemented
        'test': test,
        'sc': sc,
        'hm': hm,
        'pp': pp,
        'plot': plot,
        'clean': clean
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
        prog='Multilayer Perceptron',
        description='An artificial neural networks with multiple layers \
        to predict whether a cancer is malignant or benign on a dataset of breast cancer diagnosis.')

    parser.add_argument('--file', type=str, help='Path to the CSV file', default='data/data.csv')
    parser.add_argument('--install', action='store_true', help='Install the required packages')
    parser.add_argument('--run', action='store_true', help='Run the program')
    parser.add_argument('--random', action='store_true', help='Set random training and validation data')
    parser.add_argument('--train', action='store_true', help='Train the neural network model')
    parser.add_argument('--test', action='store_true', help='Run the tests')
    parser.add_argument('--sc', action='store_true', help='Create scatter plots')
    parser.add_argument('--hm', action='store_true', help='Create heat maps')
    parser.add_argument('--pp', action='store_true', help='Create pair plots')
    parser.add_argument('--plot', action='store_true', help='Create all plots')
    parser.add_argument('--clean', action='store_true', help='Clean up temporary files')

    return parser.parse_args()

def install() -> None:
    """Install the required packages"""
    print("Installing the required packages...")
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

def run(args: argparse.Namespace) -> None:
    """Run the program"""
    print("Running the program...")
    subprocess.run(['python3', 'src/main.py', '--file', args.file])

def train(args: argparse.Namespace) -> None:
    """Train the neural network model"""
    print("Training the neural network model...")
    if args.file:
        subprocess.run(['python3', 'src/train.py', '--file', args.file])
    else:
        print("No file specified for training. Use default training set.")

def test() -> None:
    """Run the tests"""
    print("Running the tests...")
    subprocess.run(['python3', '-m', 'pytest', 'src/tests'])

def plot(args: argparse.Namespace) -> None:
    """Create all plots"""
    import data_visualization as dv

    print("Creating all plots...")
    if args.file:
        create_dirs(['plots', 'plots/scatter_plots', 'plots/heat_maps', 'plots/pair_plots'])
        dv.plots('hm')
        dv.plots('sc')
        dv.plots('pp')
    else:
        print("No file specified for creating plots. Use default csv file.")

def hm(args: argparse.Namespace) -> None:
    """Create heat maps"""
    import data_visualization as dv
    
    print("Creating heat maps...")
    if args.file:
        create_dirs(['plots', 'plots/heat_maps'])
        dv.plots('hm')
    else:
        print("No file specified for creating heat maps. Use default csv file.")

def sc(args: argparse.Namespace) -> None:
    """Create scatter plots"""
    import data_visualization as dv
    
    print("Creating scatter plots...")
    if args.file:
        create_dirs(['plots', 'plots/scatter_plots'])
        dv.plots('sc')
    else:
        print("No file specified for creating scatter plots. Use default csv file.")

def pp(args: argparse.Namespace) -> None:
    """Create pair plots"""
    import data_visualization as dv
    
    print("Creating pair plots...")
    if args.file:
        create_dirs(['plots', 'plots/pair_plots'])
        dv.plots('pp')
    else:
        print("No file specified for creating pair plots. Use default csv file.")

def clean() -> None:
    """Clean up temporary files"""
    print("Cleaning up temporary files...")
    subprocess.run(['rm', '-rf', 'plots'])

def create_dirs(dir_list):
    """Helper function to create directories."""
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)

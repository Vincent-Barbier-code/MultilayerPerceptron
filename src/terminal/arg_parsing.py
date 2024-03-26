"""This module contains the parsing of args"""

import argparse
import subprocess
import os
import inspect
import data_visualization as dv


def execute(args):
    """Execute the command"""

    commands = {
        'install': install,
        'run': run,
        'train': train,
        'test': test,
        'sc': sc,
        'hm': hm,
        'clean': clean
    }

    for arg, func in commands.items():
        if getattr(args, arg):
            if len(inspect.signature(func).parameters) == 0:
                func()  # Appelle la fonction sans arguments
            else:
                func(args)  # Appelle la fonction avec args comme argument
            break
    
def arg_parsing():
    """Pos.systemarse the arguments"""
    parser = argparse.ArgumentParser(
        prog='Multilayer Perceptron',
        description='An artificial neural networks with multiple layers \
        to predict whether a cancer is malignant or benign on a dataset of breast cancer diagnosis.')

    parser.add_argument('--file', type=str, help='Path to the CSV file', default='data/data.csv')
    parser.add_argument('--install', action='store_true', help='Install the required packages')
    parser.add_argument('--run', action='store_true', help='Run the program')
    parser.add_argument('--train', action='store_true', help='Train the neural network model')
    parser.add_argument('--test', action='store_true', help='Run the tests')
    parser.add_argument('--sc', action='store_true', help='Create scatter plots')
    parser.add_argument('--hm', action='store_true', help='Create heat maps')
    parser.add_argument('--clean', action='store_true', help='Clean up temporary files')

    return parser.parse_args()

def install():
    """Install the required packages"""
    print("Installing the required packages...")
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])


def run(args):
    """Run the program"""
    print("Running the program...")
    subprocess.run(['python3', 'src/main.py', '--file', args.file])


def train(args):
    """Train the neural network model"""
    print("Training the neural network model...")
    if args.file:
        subprocess.run(['python3', 'src/train.py', '--file', args.file])
    else:
        os.systemprint("No file specified for training. Use default training set.")


def test():
    """Run the tests"""
    print("Running the tests...")
    subprocess.run(['python3', '-m', 'pytest', 'src/tests'])
    # subprocess.run(['python3', 'src/tests/test_data_extractor.py'])



def hm(args):
    """Create heat maps"""
    print("Creating heat maps...")
    if args.file:
        create_dirs(['plots', 'plots/heat_maps'])
        subprocess.run(['python3', 'src/data_visualization/plots.py', 'hm', '--file', args.file])
    else:
        print("No file specified for creating heat maps. Use default csv file.")


def sc(args):
    """Create scatter plots"""
    print("Creating scatter plots...")
    if args.file:
        create_dirs(['plots', 'plots/scatter_plots'])
        # subprocess.run(['python3', 'src/data_visualization/plots.py', 'sc', '--file', args.file])
        dv.plots('sc')

    else:
        print("No file specified for creating scatter plots. Use default csv file.")


def clean():
    """Clean up temporary files"""
    print("Cleaning up temporary files...")
    subprocess.run(['rm', '-rf', 'plots'])


def create_dirs(dir_list):
    """Helper function to create directories."""
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)

"""Main module"""

from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute
from data_processing.set_random import set_random_data
from data_processing import data_extractor as extractor
from perceptron.perceptron import Perceptron

import numpy as np

def main() -> None:
    """Main function"""

    args = arg_parsing()
    execute(args)
    dataframe = extractor.extract_data("data/data.csv")
    if args.random and args.run:
        train_data = set_random_data(dataframe)[0]
        validation_data = set_random_data(dataframe)[1]

    perceptron1 = Perceptron((dataframe)[1].size)
    
    perceptron1.activation((dataframe)[2])
    perceptron1.step((dataframe)[2])

if __name__ == "__main__":
    main()

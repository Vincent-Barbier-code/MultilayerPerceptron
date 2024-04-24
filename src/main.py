"""Main module"""

from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute
from data_processing.set_random import set_random_data
from data_processing import data_extractor as extractor
from perceptron.perceptron import Perceptron
from perceptron.layer import Layer

import numpy as np

def main() -> None:
    """Main function"""

    args = arg_parsing()
    execute(args)
    dataframe = extractor.extract_data("data/data.csv")

    df_features = dataframe.drop(columns=[0])


    layer1 = Layer((df_features)[2].size)
    layer1.forward((df_features)[2])
    

if __name__ == "__main__":
    main()

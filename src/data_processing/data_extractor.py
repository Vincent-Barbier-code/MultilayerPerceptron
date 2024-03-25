"""Module to extract data from a csv file"""

import numpy as np
import pandas as pd

from terminal.print import print_error


def extract_data(csv_file):
    """Extract data from a csv file"""

    try:
        data = pd.read_csv(csv_file, header=None)
    except FileNotFoundError:
        print_error(f"The file {csv_file} was not found")
        return None
    except pd.errors.EmptyDataError:
        print_error(f"The file {csv_file} is empty")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print_error(f"The file {csv_file} could not be parsed")
        return None

    return data


def column_names(i):
    """Return the column names of a DataFrame"""
    switcher = {
        0: "diagnosis",
        1: "radius",
        3: "texture",
        4: "perimeter",
        5: "area",
        6: "smoothness",
        7: "compactness",
        8: "concavity",
        9: "concave points",
        10: "symmetry",
        11: "fractal dimension",
    }
    return switcher.get(i, "other column")


def all_column_names():
    """Return all column names of a DataFrame"""
    return [
        "diagnosis",
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave points",
        "symmetry",
        "fractal dimension",
    ]


def normalize_data(data):
    """Normalize the data"""
    data = (data - data.min()) / (data.max() - data.min())
    return data


def normalize_dataframe(dataframe):
    """Normalize the dataframe"""
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        dataframe[column] = normalize_data(dataframe[column])
    return dataframe

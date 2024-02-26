"""Module to extract data from a csv file"""

import pandas as pd

from terminal_display.print import print_error


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


def extract_column(data, column_number):
    """Extract a column from a DataFrame"""
    return data.iloc[:, column_number]


def create_dataframe(data, column_number, name):
    """Create a DataFrame from IdNumber and column"""
    return pd.DataFrame(
        {"IdNumber": data.iloc[:, 0], name: data.iloc[:, column_number]}
    )


def column_names(i):
    """Return the column names of a DataFrame"""
    switcher = {
        1: "radius",
        2: "diagnosis",
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

import pandas as pd
import sys


class Extract:
    """General class for data extracting"""

    def __init__(self, csv_file, header=None):
        self.data = pd.DataFrame()
        try:
            if header:
                self.data = pd.read_csv(csv_file, header=header)
            else:
                self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            sys.exit(f"File not found {csv_file}")
        except pd.errors.EmptyDataError:
            sys.exit("Empty file")

    def standardize(self) -> pd.DataFrame:
        """Standardize all dataframe

        Returns:
            pd.DataFrame: current dataframe
        """
        self.data = (self.data - self.data.mean()) / self.data.std()
        return self.data

    def fillNaN(self, exlusions=[]) -> pd.DataFrame:
        """Fill missing values with the mean of the columns

        Args:
            exlusions (list, optional): list of features to exclude

        Returns:
            pd.DataFrame: current dataframe
        """
        keys = set(self.data.columns.values).difference(exlusions)
        for key in keys:
            self.data.fillna(value={key: self.data[key].mean()}, inplace=True)
        return self.data

    def dropColumns(self, col: list[str]) -> pd.DataFrame:
        """Drop given list of columns in dataframe

        Args:
            col (list[str]): list of columns to drop

        Returns:
            pd.DataFrame: current dataframe
        """
        self.data = self.data.drop(col, axis=1)
        return self.data

    def get_data_training(
        self, label_name: str, need_facto: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get x, y for training purpose

        Args:
            label_name (str): label to predict
            need_facto (bool, optional): is label need to be factorized

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: data standadized (x), labels (y)
        """
        x = self.data.drop(label_name, axis=1)
        y = pd.DataFrame()
        if need_facto:
            y = pd.DataFrame(pd.factorize(self.data[label_name])[0])
        else:
            y = pd.DataFrame(self.data[label_name])
        x = (x - x.mean()) / x.std()
        return x, y

    def get_data_test(self) -> pd.DataFrame:
        x = self.data
        x = (x - x.mean()) / x.std()
        return x

"""Module to create plots"""

import matplotlib.pyplot as plt
from alive_progress import alive_bar
from data_processing import extract as extract
import seaborn as sns
import numpy as np

class plot:
    """Class to create plots"""

    def __init__(self, data: np.ndarray) -> None:
        """Initialize the data"""
        self.data = data

    def plot(self) -> None:
        """Plot the data"""

        with alive_bar(100) as bar:
            sns.pairplot(self.data.data)
            bar()

        plt.show()

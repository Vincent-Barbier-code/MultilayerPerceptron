"""Module to create plots"""

import matplotlib.pyplot as plt
# from alive_progress import ap
from data_processing import extract as extract
import seaborn as sns

class Plot:
    """Class to create plots"""

    def __init__(self, epochs, losses, val_losses) -> None:
        """Initialize the data"""
        self.epochs = epochs
        self.losses = losses
        self.val_losses = val_losses


    def plots(self) -> None:
        """Plot the data"""
        self.loss_plot()
    
    def loss_plot(self) -> None:
        """Plot the neural network."""

        lineplot = sns.lineplot(x=self.epochs, y=self.losses, color="red", label="train")
        sns.lineplot(x=self.epochs, y=self.val_losses, color="blue", label="test")
        lineplot.set(xlabel="Epochs", ylabel="Loss")
        fig = lineplot.get_figure()
        plt.show()

    
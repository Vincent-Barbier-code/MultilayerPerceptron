"""Module to create plots"""

import matplotlib.pyplot as plt
# from alive_progress import ap
from data_processing import extract as extract
import seaborn as sns

class Plot:
    """Class to create plots"""

    def __init__(self, losses, val_losses) -> None:
        """Initialize the data"""
        self.losses = losses
        self.val_losses = val_losses


    def plots(self) -> None:
        """Plot the data"""
        self.loss_plot()
    
    def loss_plot(self) -> None:
        """Plot the network network."""

        epochs = range(0, len(self.losses))
        min_val_loss = min(self.val_losses)
        min_val_loss_index = self.val_losses.index(min_val_loss)
        lineplot = sns.lineplot(x=epochs, y=self.losses, color="red", label="train")
        sns.lineplot(x=epochs, y=self.val_losses, color="blue", label="test")
        plt.plot(epochs[min_val_loss_index], min_val_loss, color="red", marker="o", markersize=5)
        plt.text(epochs[min_val_loss_index], min_val_loss, f"min loss: {min_val_loss:.2f}", fontsize=12)
        lineplot.set(xlabel="Epochs", ylabel="Loss")

        # fig = lineplot.get_figure()
        plt.show()

    
"""Module to create plots"""

import matplotlib.pyplot as plt
import seaborn as sns

# from alive_progress import ap
from data_processing import extract as extract
from perceptron.network import Network


def loss_plot(network: Network) -> None:
    """Plot the network network."""
    losses = network.losses
    val_losses = network.val_losses

    epochs = range(0, len(losses))
    min_val_loss = min(val_losses)
    min_val_loss_index = val_losses.index(min_val_loss)
    lineplot = sns.lineplot(x=epochs, y=losses, color="red", label="train")
    sns.lineplot(x=epochs, y=val_losses, color="blue", label="test")
    plt.plot(epochs[min_val_loss_index], min_val_loss, color="red", marker="o", markersize=5)
    plt.text(epochs[min_val_loss_index], min_val_loss, f"min loss: {min_val_loss:.2f}", fontsize=12)
    lineplot.set(xlabel="Epochs", ylabel="Loss")

    # fig = lineplot.get_figure()
    plt.show()

def bench_plots(networks: list[Network]) -> None:
    """Plot the benchmark"""
    
    fig, ax = plt.subplots()
    for network, name in zip(networks, ["SGD", "Momentum", "Adam", "Rmsprop"]):
        ax.plot(range(0, len(network.losses)), network.losses, label=name)
    ax.set(xlabel="Epochs", ylabel="Loss")
    ax.legend()
    plt.show()
        

    
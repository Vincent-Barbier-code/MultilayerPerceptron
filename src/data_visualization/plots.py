"""Module to create plots"""

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from data_processing import extract as extract
from perceptron.network import Network


def plot_data(data: pd.DataFrame) -> None:
    """Plot the data"""

    data = data.drop(data.columns[0], axis=1)
    data.drop(data.columns[range(11, data.shape[1])], axis=1, inplace=True)
    columns_names = [
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
        "fractal_dimension",
    ]
    data.columns = columns_names
    sns.pairplot(data, hue=data.columns[0])
    plt.show()


def loss_acc_plot(network: Network) -> None:
    """Plot the network losses and accuracy"""

    losses = network.losses
    val_losses = network.val_losses
    accuracy = network.accuracies
    val_accuracy = network.val_accuracies

    epochs = range(0, len(losses))
    min_val_loss = min(val_losses)
    min_val_loss_index = val_losses.index(min_val_loss)
    fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
    fig.suptitle("Loss and Accuracy")

    axs[0].plot(epochs, losses, color="red", label="train")
    axs[0].plot(epochs, val_losses, color="blue", label="validation")
    axs[0].plot(
        [min_val_loss_index],
        [min_val_loss],
        marker="o",
        markersize=5,
        color="blue",
        label=f"min val_loss: {min_val_loss:.2f}",
    )
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(epochs, accuracy, color="green", label="train accuracy")
    axs[1].legend()
    axs[1].plot(epochs, val_accuracy, color="orange", label="validation accuracy")
    axs[1].legend()
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")

    plt.show()


def bench_plots(networks: list[Network]) -> None:
    """Plot the benchmark"""

    fig, ax = plt.subplots()
    for network, name in zip(networks, ["SGD", "Momentum", "Adam", "Rmsprop"]):
        ax.plot(range(0, len(network.losses)), network.losses, label=name)
    ax.set(xlabel="Epochs", ylabel="Loss")
    ax.legend()

    plt.show()


def metrics_plot(network: Network):
    """Plot the metrics"""

    fig, axs = plt.subplots(3, figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle("Metrics")

    axs[0].plot(range(0, len(network.f1_scores)), network.f1_scores, label="f1_score")
    axs[0].set(xlabel="Epochs", ylabel="F1 score")

    axs[1].plot(range(0, len(network.recalls)), network.recalls, label="recall")
    axs[1].set(xlabel="Epochs", ylabel="Recall")

    axs[2].plot(
        range(0, len(network.precisions)), network.precisions, label="precision"
    )
    axs[2].set(xlabel="Epochs", ylabel="Precision")

    plt.show()

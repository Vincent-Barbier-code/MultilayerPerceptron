"""Module to create plots"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# from alive_progress import ap
from data_processing import extract as extract
from perceptron.network import Network

def plot_data(data: pd.DataFrame) -> None:
    """Plot the data"""
    

    data = data.drop(data.columns[0], axis=1)
    data.drop(data.columns[range(11, data.shape[1])], axis=1, inplace=True)
    columns_names = ["diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal_dimension"]
    data.columns = columns_names
    sns.pairplot(data, hue=data.columns[0])
    plt.show()

def loss_plot(network: Network) -> None:
    """Plot the network losses"""

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
    
    fig, axs = plt.subplots(4, figsize=(10, 10), sharex=True, sharey=True)  
    fig.suptitle("Metrics")
    
    axs[0].plot(range(0, len(network.f1_scores)), network.f1_scores, label="f1_score")
    axs[0].set(xlabel="Epochs", ylabel="F1 score")

    axs[1].plot(range(0, len(network.accuracies)), network.accuracies, label="accuracy")
    axs[1].set(xlabel="Epochs", ylabel="Accuracy")

    axs[2].plot(range(0, len(network.recalls)), network.recalls, label="recall")
    axs[2].set(xlabel="Epochs", ylabel="Recall")

    axs[3].plot(range(0, len(network.precisions)), network.precisions, label="precision")
    axs[3].set(xlabel="Epochs", ylabel="Precision")

    plt.show()

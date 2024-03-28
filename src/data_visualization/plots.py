"""Module to create plots"""

import matplotlib.pyplot as plt
from alive_progress import alive_bar
import data_processing.data_extractor as extractor

def plots(args):
    """Create different plots from a DataFrame"""
    csv_file = "data/data.csv"
    extracted_data = extractor.extract_data(csv_file)
    if extracted_data is None:
        return

    dataframe_normalized = extractor.normalize_dataframe(extracted_data)
    dataframe_normalized[1] = dataframe_normalized[1].map({"M": 1, "B": 0})

    if args == "sc":
        create_scatter_plots(dataframe_normalized)
    elif args == "hm":
        heat_map(dataframe_normalized)
    else:
        print("Invalid argument")


def scatter_plot(dataframe, x, y, z):
    """Create a scatter plot from a DataFrame"""
    x_column = extractor.column_names(x)
    y_column = extractor.column_names(y)
    colors = dataframe[1]
    plt.scatter(x=dataframe[x], y=dataframe[y], c=colors)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.savefig("plots/scatter_plots/scaterplot" + str(z) + ".png")
    plt.clf()


def create_scatter_plots(dataframe):
    """Create scatter plots from a DataFrame"""
    z = 0
    total_plots = 36

    with alive_bar(total_plots, bar="smooth", title="scatter_plots") as load_bar:
        for i in range(2, 11):
            for y in range(i + 1, 11):
                z += 1
                scatter_plot(dataframe, i, y, z)
                load_bar()  # pylint: disable=E1102


def heat_map(dataframe):
    """Create a heat map from a DataFrame"""
    dataframe = dataframe.drop(columns=[0])
    dataframe = dataframe.drop(columns=range(12, 32))
    corr_matrix = dataframe.corr()

    fig, ax = plt.subplots(figsize=(7, 7))  # Créez une figure et des axes
    cax = ax.matshow(
        corr_matrix, cmap="viridis"
    )  # Utilisez ax.matshow au lieu de plt.imshow
    fig.colorbar(cax)

    plt.title("Correlation between Diagnosis and Features")

    # Utilisez plt.xticks et plt.yticks pour définir les labels des axes
    plt.xticks(ticks=range(corr_matrix.shape[1]), labels=extractor.all_column_names())
    plt.yticks(ticks=range(corr_matrix.shape[0]), labels=extractor.all_column_names())

    ax.xaxis.tick_bottom()

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Afficher les valeurs de corrélation à l'intérieur de la heatmap
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
            )
    plt.tight_layout()
    plt.savefig("plots/heat_maps/heatmap.png")
    plt.clf()

"""
Main script to run to generate heatmap data and plot the results.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data import constants
from experiments.heatmap.generate_new import generate_rf_heatmap_data
from experiments.heatmap.generate_old import generate_linreg_nn_data

FIGURE_DIR = "experiments/figures/eds-revisions"
LINREG_DATA_PATH = "experiments/heatmap/linreg_heatmap_data.npy"
NN_DATA_PATH = "experiments/heatmap/nn_heatmap_data.npy"
RF_DATA_PATH = "experiments/heatmap/rf_heatmap_data.npy"


def plot_heatmap(ax: plt.Axes, data: np.ndarray, scale: dict = None, title: str = None) -> plt.Axes:
    """
    Plots a heatmap given a 2d numpy array.
    """
    idxs = [constants.LAND_USE_COLS.index(col) for col in constants.RECO_COLS]
    non_idxs = [constants.LAND_USE_COLS.index(col) for col in constants.LAND_USE_COLS if col not in constants.RECO_COLS]
    sorted_labels = np.array(constants.LAND_USE_COLS)[idxs + non_idxs]

    ax = sns.heatmap(data,
                     center=0,
                     cmap="PiYG_r",
                     xticklabels=constants.RECO_COLS,
                     yticklabels=sorted_labels,
                     ax=ax,
                     cbar=False,
                     **scale)
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    return ax


def main():
    """
    Main function to generate and plot heatmaps.
    Generates heatmap data if it doesn't exist, then plots it.
    """
    if not os.path.exists(LINREG_DATA_PATH) or not os.path.exists(NN_DATA_PATH):
        print("Generating old heatmap data...")
        generate_linreg_nn_data()
    if not os.path.exists(RF_DATA_PATH):
        print("Generating new heatmap data...")
        generate_rf_heatmap_data()

    print("Loading heatmap data from disk...")
    nn_data = np.load(NN_DATA_PATH)
    rf_data = np.load(RF_DATA_PATH)
    linreg_data = np.load(LINREG_DATA_PATH)

    print("Plotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes = axes.flatten()

    scale = {"vmin": -155, "vmax": 155}
    axes[0] = plot_heatmap(axes[0], nn_data, title="a) Neural Net Predictions", scale=scale)
    axes[1] = plot_heatmap(axes[1], rf_data, title="b) RF Predictions", scale=scale)
    axes[2] = plot_heatmap(axes[2], linreg_data, title="c) LinReg Predictions", scale=scale)

    fig.supxlabel("To")
    fig.supylabel("From")
    fig.colorbar(axes[0].collections[0], ax=axes, orientation='vertical', label="ELUC (tC/ha)")

    plt.savefig(os.path.join(FIGURE_DIR, "heatmap-fig.png"), dpi=300)

    plt.show()


if __name__ == "__main__":
    main()

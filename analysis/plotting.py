import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def annotate_multiple_ecdf(datasets, labels, x_offset=None, y_offset=0.0, ha=None, levels=None, colors=None, linestyles=None, ax=None):
    """
    Annotate multiple ECDF plots with one annotation per line.

    Parameters:
    datasets (list of array-like): List of datasets for the ECDF plots.
    labels (list of str, optional): List of labels for each dataset.
    """
    if ax is None:
        ax = plt.gca()

    if levels is None:
        levels = [80] * len(labels)

    if linestyles is None:
        linestyles = ["-"] * len(labels)

    if colors is None:
        colors = [None] * len(labels)

    if ha is None:
        ha = ["left"] * len(labels)

    if x_offset is None:
        x_offset = [0] * len(labels)

    for i, data in enumerate(datasets):
        # Create the ECDF plot for each dataset
        l1 = sns.ecdfplot(data, linestyle=linestyles[i], color=colors[i], ax=ax)

        # Get the ECDF data points
        ecdf_data = np.sort(data)
        ecdf_y = np.arange(1, len(data) + 1) / len(data)

        # Select a middle x-coordinate for annotation
        x_coord = np.percentile(data, levels[i])
        y_coord = np.interp(x_coord, ecdf_data, ecdf_y)

        # Add the annotation
        ax.annotate(f'{labels[i]}',
                     xy=(x_coord + x_offset[i], y_coord + y_offset), color=l1.lines[-1].get_color(), ha=ha[i])

def scaled_kde(data, desired_peak=1, npts=1000):
    # Sample data
    # Compute KDE using scipy
    kde = gaussian_kde(data)

    # Generate x values for the KDE plot
    x_values = np.linspace(min(data), max(data), npts)
    y_values = kde(x_values)

    # Find the peak of the KDE
    peak_value = np.max(y_values)
    # Desired peak value

    # Scale the y_values so the peak matches the desired value
    scaled_y_values = (y_values / peak_value) * desired_peak

    return x_values, scaled_y_values
import matplotlib
import numpy as np
import torch

matplotlib.use("agg")
import matplotlib.pyplot as plt


def np2torch(x):
    x = torch.from_numpy(x)
    if x.dtype == torch.float64:
        x = x.to(torch.float32)
    return x


def standard_error(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))


def export_plot(ys, ylabel, title, filename):
    """
    Export a plot in filename

    Args:
        ys: (list) of float / int to plot
        filename: (string) directory
    """
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

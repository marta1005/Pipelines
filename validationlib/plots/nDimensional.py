'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as st
from scipy.sparse import csr_matrix
import scipy.cluster.hierarchy as sch

from validationlib.misc.subsampling import maskApply

def scatterplotMatrix(
    data: np.ndarray,
    names: List[str],
    method: str,
    nbins: int = 12,
    kdeBins: int = 4,
    s: float = 1,
    colornorm: Callable = mcolors.LogNorm(),
    figsize: float = 8
) -> plt.figure:
    """
    Creates a matrix of scatterplots, 2D histograms, or 2D histograms with KDE.

    :param data: Numpy array of data.
    :param names: List of names for variables.
    :param method: 'scatter', '2dhist', or '2dhistKDE'.
    :param nbins: Number of bins for histograms (default=12).
    :param kdeBins: Number of bins for KDE (default=4).
    :param s: Size of points in scatterplot (default=1).
    :param colornorm: Color normalization function (default=LogNorm).
    :param figsize: Size of the figure (default=8).
    :return: Matplotlib figure.
    """    
    numdata, numvars = data.shape

    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(figsize, figsize))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Plot the data.
    for row in range(numvars):
        for col in range(numvars):
            if row == col:
                continue

            if method == "scatter":
                axes[row, col].scatter(data[:, col], data[:, row], s=s)
            elif method == "2dhist":
                Z, yedges, xedges = np.histogram2d(data[:, row], data[:, col], bins=nbins, density=True)
                axes[row, col].pcolormesh(xedges, yedges, Z, norm=colornorm, cmap=plt.cm.jet)
            elif method == "2dhistKDE":
                Z, yedges, xedges = np.histogram2d(data[:, row], data[:, col], bins=kdeBins)

                xdata, ydata, frequency = [], [], []
                for m in range(kdeBins):
                    xCenter = (xedges[m + 1] + xedges[m]) / 2
                    for n in range(kdeBins):
                        yCenter = (yedges[n + 1] + yedges[n]) / 2
                        xdata.append((xedges[m + 1] + xedges[m]) / 2)
                        ydata.append((yedges[n + 1] + yedges[n]) / 2)
                        frequency.append(Z[m, n])

                xy = np.vstack([xdata, ydata])
                k = st.gaussian_kde(xy, weights=frequency)

                Z, yedges, xedges = np.histogram2d(data[:, row], data[:, col], bins=nbins)
                for m in range(nbins):
                    xCenter = (xedges[m + 1] + xedges[m]) / 2
                    for n in range(nbins):
                        yCenter = (yedges[n + 1] + yedges[n]) / 2
                        Z[m, n] = k([xCenter, yCenter])

                axes[row, col].pcolormesh(xedges, yedges, Z, cmap=plt.cm.jet)

    # Set up ticks only on one side for the "edge" subplots, remove the rest
    for ax in axes.flat:
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        if ax.is_first_col() and ax.is_last_row():
            continue
        elif ax.is_first_col():
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        elif ax.is_last_row():
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        else:
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords="axes fraction", ha="center", va="center")
        axes[i, i].xaxis.set_visible(False)
        axes[i, i].yaxis.set_visible(False)

    return fig

def catHeatmap(
    df: pd.DataFrame,
    errordf: pd.DataFrame,
    id_columns: List[str],
    id_1_order: List[str],
    id_2_order: List[str],
    mapdisplay: str = "Percentage Error(%)",
    criteria: str = "crit",
    annot: bool = False,
    annotfontsize: int = 8,
    figsize: float = 16,
    cmap: Callable = plt.cm.coolwarm_r
):
    """
    Creates a heatmap of a metric for combinations of categories using criteria.

    :param df: DataFrame containing categorical variables.
    :param errordf: DataFrame with error metrics.
    :param id_columns: Names of the categories.
    :param id_1_order: Order of the first category.
    :param id_2_order: Order of the second category.
    :param mapdisplay: Metric to display on the heatmap (default="Percentage Error(%)").
    :param criteria: Criteria to select the metric (default="crit").
    :param annot: Display text annotations inside the heatmap (default=False).
    :param annotfontsize: Font size of annotations (default=8).
    :param figsize: Size of the figure (default=16).
    :param cmap: Colormap for the heatmap (default=plt.cm.coolwarm_r).
    :return: Summary report DataFrame.
    """
    if len(id_columns) != 2:
        raise ValueError(f"Id columns are not properly defined. Two columns required and passed: {id_columns}")
    cat1, cat2 = id_columns[0], id_columns[1]
    nrows, ncols = len(id_1_order), len(id_2_order)
    matrix = np.full((nrows, ncols), np.nan)

    temp = pd.DataFrame(df[cat1][:])
    temp[cat2] = df[cat2][:]
    temp["lc"] = df["lc"][:]
    temp["Failure Mode"] = errordf[criteria + "_RFname"][:]
    temp["True Value"] = errordf[criteria + "_RF"][:]
    temp["Predicted Value"] = errordf[criteria + "_RF"][:] - errordf[criteria + "_res"][:]
    temp["Residual"] = errordf[criteria + "_res"][:]
    temp["Percentage Error(%)"] = errordf[criteria + "_percent"]

    if mapdisplay == "True Value" or mapdisplay == "Predicted Value":
        norm = mcolors.TwoSlopeNorm(vcenter=1)
    elif mapdisplay == "Failure Mode":
        RFtypes = sorted(list(pd.unique(temp[mapdisplay].ravel())))
        n = len(RFtypes)
        norm, cmap = None, sn.color_palette("Pastel2", n)
    else:
        norm = mcolors.TwoSlopeNorm(vcenter=0)

    # Get the specified value 'mapdisplay' from the critical pair of categories
    indexes = []
    for name, group in temp.groupby(by=[cat1, cat2]):
        if criteria == "crit":
            idx = group["True Value"].idxmin(axis=0)
        elif criteria == "maxpercent":
            idx = group["Percentage Error(%)"].abs().idxmax(axis=0)
        row, col = id_1_order.index(name[0]), id_2_order.index(name[1])
        indexes.append(idx)

        if mapdisplay == "Failure Mode":
            matrix[row, col] = RFtypes.index(temp.at[idx, mapdisplay])
            continue
        matrix[row, col] = round(temp.at[idx, mapdisplay], 3)

    summary_report = temp.iloc[indexes, :]

    # Plot
    plt.figure(figsize=(figsize, figsize))

    heatmap = sn.heatmap(
        matrix,
        linewidths=0.1,
        annot=annot,
        xticklabels=1,
        yticklabels=1,
        norm=norm,
        cmap=cmap,
        annot_kws={"size": annotfontsize},
    )
    plt.title(
        mapdisplay + " with -" + criteria + "- criteria",
        fontsize="large",
        weight="semibold"
    )

    if cat1 == "Stringer":
        id_1_order = [elem.split("Str")[1] for elem in id_1_order]
    heatmap.set_yticklabels(id_1_order)
    plt.ylabel(cat1, fontsize="large", weight="semibold")
    heatmap.set_xticklabels(id_2_order)
    plt.xlabel(cat2, fontsize="large", weight="semibold")

    if mapdisplay == "Failure Mode":
        colorbar = heatmap.collections[0].colorbar
        r = colorbar.vmax - colorbar.vmin
        colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(RFtypes)

    return summary_report

def clusteredCorrHeatmap(
    dataframe: pd.DataFrame,
    method: str = "pearson",
    annot: bool = False
):
    """
    Creates a clustered correlation heatmap.

    :param dataframe: DataFrame.
    :param method: Correlation method ('pearson', 'kendall', or 'spearman', default='pearson').
    :param annot: Display text annotations inside the heatmap (default=False).
    :return: Clustered correlation matrix.
    """
    corr_matrix = dataframe.corr(method=method)
    pairwise_distances = sch.distance.pdist(corr_matrix)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 10
                                                       
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    if isinstance(corr_matrix, pd.DataFrame):
        clustered_corr = corr_matrix.iloc[idx, :].T.iloc[idx, :]
    else:
        clustered_corr = corr_matrix[idx, :][:, idx]

    plotsize = len(dataframe.columns) * 0.8
    _ = plt.figure(figsize=(plotsize, plotsize))
    sn.heatmap(clustered_corr, linewidths=0.1, annot=annot, xticklabels=1, yticklabels=1,
               vmax=1, vmin=-1)
    plt.title("Clustered Correlation Matrix ("+method+")")

    return clustered_corr

def corrHeatmap(
    dataframe: pd.DataFrame,
    mask: Optional[csr_matrix] = None,
    method: str = "pearson",
    annot: bool = False
) -> plt.figure:
    """
    Creates a correlation heatmap.

    :param dataframe: DataFrame.
    :param mask: Optional mask for filtering.
    :param method: Correlation method ('pearson', 'kendall', or 'spearman', default='pearson').
    :param annot: Display text annotations inside the heatmap (default=False).
    :return: Matplotlib figure.
    """
    
    plotsize = len(dataframe.columns) * 0.8
    fig = plt.figure(figsize=(plotsize, plotsize))

    data = maskApply(dataframe, mask, columnIndex='all', to_numpy=False)

    corr2D = data.corr(method=method)
    sn.heatmap(corr2D, linewidths=0.1, annot=annot, xticklabels=1, yticklabels=1,
               vmax=1, vmin=-1)
    plt.title("Correlation matrix (" + str(method) + ")")

    return fig

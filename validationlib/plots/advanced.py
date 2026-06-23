'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib as mlp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as pltick
import matplotlib.patheffects as patheffects
import scipy.stats as st
from scipy.sparse import csr_matrix

from .common import multiplePlots, histnbins
from ..misc.subsampling import maskApply, asign_binnedIndices, kPartitions_equi, bin_data
from ..misc.utils import check_column_number

def catBoxplot(
    df: pd.DataFrame,
    errordf: pd.DataFrame,
    category: str,    
    mask: csr_matrix = None,
    categoryorder: Optional[list] = None,
    errortype: str = "Metric",
    figure_length: Union[float, str] = 'auto',
    nobs: bool = True,
    showfliers: bool = True,
    swarmplot: bool = False
) -> plt.figure:
    """
    Deprecated. Use boxplot instead.
    Plot a categorical boxplot with optional swarmplot overlay.

    :param df: Input DataFrame.
    :param errordf: Error DataFrame.
    :param mask: Masking matrix.
    :param category: Column name for categorical variable.
    :param categoryorder: Optional order of categories.
    :param errortype: Type of error. Defaults to "Metric".
    :param figure_length: Length of the figure. Defaults to 'auto'.
    :param nobs: Display number of observations. Defaults to True.
    :param showfliers: Display outliers. Defaults to True.
    :return: Matplotlib figure.
    """

    data = maskApply(df, mask, columnIndex='all', to_numpy=False)
    data = data[[category]]

    data_err = maskApply(errordf, mask, columnIndex='all', to_numpy=False)

    if categoryorder != None:
        order = categoryorder.copy()
    else:
        order = None

    if order != None:
        order = [cat for cat in order if (data[category] == cat).any()]

    if figure_length == 'auto':
        figure_length = max(len(data.squeeze().unique())*0.2, 8)

    plt.figure(figsize=(figure_length, 8))
    sn.boxplot(
        x=data.squeeze(),
        y=data_err.squeeze(),
        color="c",
        order=order,
        showfliers=showfliers,
    )
    plt.xticks(rotation='vertical')
    if swarmplot:
        sn.swarmplot(x=category, y=errordf.columns[0], data=pd.concat([data,data_err]), color=".25", order=order)

    plt.ylabel(errortype)
    plt.title(errortype + " vs " + category + ": n" + str(len(data)) + " observations")

    # Set the y-axis tick label formatting
    ax = plt.gca()
    ax.yaxis.set_major_formatter(pltick.FormatStrFormatter('%.4f'))

    # Text labels with the number of observations per category
    if nobs == True:
        if order != None:
            nobs = ["n" + str((data[category] == cat).sum()) for cat in order]

        else:
            nobs = data[category].value_counts().values
            nobs = [str(x) for x in nobs.tolist()]
            nobs = [" n" + i for i in nobs]

        ax = plt.gca()
        pos = range(len(nobs))
        for tick, _ in zip(pos, ax.get_xticklabels()):
            txt = ax.text(
                pos[tick],
                (ax.get_ylim()[0]),
                nobs[tick],
                horizontalalignment="center",
                fontsize=8,
                color="black",
                weight="semibold",
                rotation="horizontal"
            )
            txt.set_path_effects([patheffects.withStroke(linewidth=5, foreground='w')])
    return ax.get_figure()


def boxplot(
    x: pd.DataFrame,
    y: Union[pd.DataFrame, List[pd.DataFrame]],
    categorical: bool = False,
    bins: Optional[Union[int, str, list]] = None,
    show_n_obs: bool = True,
    showfliers: bool = True,
    swarmplot: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    var_y: str = None,
    hue_labels: Optional[List[str]] = None,
    mask: Optional[csr_matrix] = None,
    trimStds: Optional[float] = None,
    multiPlotsKwargs: dict={}
) -> plt.figure:
    """
    Generate a violin plot.

    :param x: Input DataFrame for x-axis.
    :param y: Input DataFrame for y-axis.
    :param bins: Union[str, int, list], optional
        Specification of the bins. Accepted values are:
        - String: Automatic binning methods accepted by numpy.histogram_bin_edges (see numpy docs).
        - Integer: Number of bins to use.
        - List: List of bin edges.
        - None: In case of categorical data (categorical=True), the unique categories will be used as bins.
    :param xlabel: Label for x-axis.
    :param ylabel: Label for y-axis.
    :param var_y: Variable name for y-axis. Defaults to None.
    :param mask: Optional masking matrix.
    :param categorical: Categorical data. Defaults to False.
    :param showextrema: Show outliers. Defaults to True.
    :param trimStds: Standard deviation for trimming data. Defaults to None.
    :param multiPlotsKwargs: Additional keyword arguments for multiple plots.
    :return: Matplotlib figure.
    """

    x, y, _, _ = check_column_number(x, y)

    if not categorical: assert bins is not None, "Bins must be specified for non-categorical data."

    trimMsg = '' if trimStds is None else fr'inside $\mu \pm {trimStds}\sigma$'
    if var_y != None:
        figTitle=f'{var_y} \n Boxplot [{xlabel} vs {ylabel}] {trimMsg}'
    else:
        figTitle = f'Boxplot [{xlabel} vs {ylabel}] {trimMsg}'

    if hasattr(x, 'shape'):
        nplots = x.shape[1]
    else:
        nplots = 1

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        dataY = [maskApply(y[j], mask, i) for j in range(len(y))] if isinstance(y, list) else [maskApply(y, mask, i)]
        bin_boundaries, binned_indices = bin_data(dataX, bins, categorical=categorical)

        if isinstance(x, pd.DataFrame):
            label = x.columns[i]
        else:
            label = x

        if categorical:
            data_dict_list = [{cat: dataY[j][binned_indices==i] for i, cat in enumerate(bin_boundaries)} for j in range(len(dataY))]
        else:
            data_dict_list = [{f"[{bin_boundaries[i]:.3g}, {bin_boundaries[i+1]:.3g}]": dataY[j][binned_indices==i] for i in range(len(bin_boundaries)-1) if sum(binned_indices==i) > 0} for j in range(len(dataY))]


        x_values = list(data_dict_list[0].keys())
        xticks = np.arange(len(x_values))*1.5
        nhues = len(data_dict_list)
        widths = 1/nhues
        offsets = np.linspace(-widths/2, widths/2, nhues)*(nhues-1)

        if trimStds is not None:
            for data_dict in data_dict_list:
                for val in (x_values):
                    if len(data_dict[val]) == 0: continue
                    else:
                        mean, stds = np.mean(data_dict[val]), np.std(data_dict[val])
                        data_dict[val] = np.clip(data_dict[val], mean - trimStds * stds, mean + trimStds * stds)

        for j, data_dict in enumerate(data_dict_list):
            color = mlp.cm.get_cmap("tab10").colors[j]
            colorprops = {
                "boxprops": {"color": color},
                "medianprops": {"color": color},
                "whiskerprops": {"color": color},
                "capprops": {"color": color},
                "flierprops": {"markerfacecolor": color, "markeredgecolor": color, "alpha": 0.9, "markersize": 2},
            }

            hue_label = hue_labels[j] if hue_labels is not None else None
            axes.boxplot(list(data_dict.values()), positions=xticks+offsets[j], widths=widths*0.9, **colorprops, showfliers=showfliers, label=hue_label)
            if swarmplot:
                sn.swarmplot(x=x_values, y=list(data_dict.values()), color=".25", ax=axes)

        axes.set_xticks(ticks=xticks, labels=x_values, rotation="vertical")
        axes.set_title(label)
        colorbarData = None

        if hue_labels is not None:
            axes.legend()

        ylim = axes.get_ylim()
        axes.set_ylim(ylim[0] - 0.1 * (ylim[1] - ylim[0]), ylim[1] + 0.1 * (ylim[1] - ylim[0]))

        # Text labels with the number of observations per category (only if one set of data)
        if show_n_obs == True and len(data_dict_list) == 1:
            nobs = [" n"+str(len(data_dict[val])) for val in x_values]

            for i, tick in enumerate(xticks):
                txt = axes.text(
                    tick,
                    (axes.get_ylim()[0]),
                    nobs[i],
                    horizontalalignment="center",
                    fontsize=8,
                    color="black",
                    weight="semibold",
                    rotation="horizontal"
                )
                txt.set_path_effects([patheffects.withStroke(linewidth=5, foreground='w')])

        return axes, colorbarData

    fig = multiplePlots(
        nplots,
        func,
        figTitle=figTitle,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=None,
        xticks_rotation='vertical',
        **multiPlotsKwargs
    )

    return fig


def violinPlot(  
    x: pd.DataFrame, 
    y: Union[pd.DataFrame, List[pd.DataFrame]],    
    bins: Optional[Union[int, str, list]] = None,
    xlabel: str = "",
    ylabel: str = "",    
    var_y: str = None,
    hue_labels: Optional[List[str]] = None,
    mask: Optional[csr_matrix] = None,
    categorical: bool = False,
    showextrema: bool = True,
    trimStds: Optional[float] = None,
    multiPlotsKwargs: dict={}
) -> plt.figure:
    """
    Generate a violin plot.

    :param x: Input DataFrame for x-axis.
    :param y: Input DataFrame for y-axis.
    :param bins: Union[str, int, list], optional
        Specification of the bins. Accepted values are:
        - String: Automatic binning methods accepted by numpy.histogram_bin_edges (see numpy docs).
        - Integer: Number of bins to use.
        - List: List of bin edges.
        - None: In case of categorical data (categorical=True), the unique categories will be used as bins.
    :param xlabel: Label for x-axis.
    :param ylabel: Label for y-axis.
    :param var_y: Variable name for y-axis. Defaults to None.
    :param mask: Optional masking matrix.
    :param categorical: Categorical data. Defaults to False.
    :param showextrema: Show outliers. Defaults to True.
    :param trimStds: Standard deviation for trimming data. Defaults to None.
    :param multiPlotsKwargs: Additional keyword arguments for multiple plots.
    :return: Matplotlib figure.
    """

    x, y, _, _ = check_column_number(x, y)

    if not categorical: assert bins is not None, "Bins must be specified for non-categorical data."

    trimMsg = '' if trimStds is None else fr'inside $\mu \pm {trimStds}\sigma$'
    if var_y != None: 
        figTitle=f'{var_y} \n Violin plot [{xlabel} vs {ylabel}] {trimMsg}'
    else: 
        figTitle = f'Violin plot [{xlabel} vs {ylabel}] {trimMsg}'    

    if hasattr(x, 'shape'):
        nplots = x.shape[1]
    else:
        nplots = 1

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)   
        dataY = [maskApply(y[j], mask, i) for j in range(len(y))] if isinstance(y, list) else [maskApply(y, mask, i)]
        bin_boundaries, binned_indices = bin_data(dataX, bins, categorical=categorical)

        if isinstance(x, pd.DataFrame):
            label = x.columns[i]
        else: 
            label = x

        if categorical:
            data_dict_list = [{cat: dataY[j][binned_indices==i] for i, cat in enumerate(bin_boundaries)} for j in range(len(dataY))]
        else:
            data_dict_list = [{f"[{bin_boundaries[i]:.3g}, {bin_boundaries[i+1]:.3g}]": dataY[j][binned_indices==i] for i in range(len(bin_boundaries)-1) if sum(binned_indices==i) > 0} for j in range(len(dataY))]


        x_values = list(data_dict_list[0].keys())
        xticks = np.arange(len(x_values))*1.5
        nhues = len(data_dict_list)
        widths = 1/nhues
        offsets = np.linspace(-widths/2, widths/2, nhues)*(nhues-1) 

        if trimStds is not None:
            for data_dict in data_dict_list:
                for val in (x_values):
                    if len(data_dict[val]) == 0: continue
                    else:
                        mean, stds = np.mean(data_dict[val]), np.std(data_dict[val])
                        data_dict[val] = np.clip(data_dict[val], mean - trimStds * stds, mean + trimStds * stds)

        legend_labels = []
        for j, data_dict in enumerate(data_dict_list):
            color = mlp.cm.get_cmap("tab10").colors[j]

            hue_label = hue_labels[j] if hue_labels is not None else None

            violin_elements = axes.violinplot(list(data_dict.values()), positions=xticks+offsets[j], widths=widths*0.9, showextrema=showextrema)
            for v_elem in violin_elements['bodies']:
                v_elem.set_edgecolor(color)
                v_elem.set_alpha(0.3)
                v_elem.set_facecolor(color)

            if hue_label is not None:
                legend_labels.append((mpatches.Patch(color=color), hue_label))

        if hue_labels is not None:
            axes.legend(*zip(*legend_labels))

        axes.set_xticks(ticks=xticks, labels=x_values, rotation="vertical")
        axes.set_title(label)
        colorbarData = None

        return axes, colorbarData

    fig = multiplePlots(
        nplots,
        func,
        figTitle=figTitle,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=None,
        xticks_rotation='vertical',
        **multiPlotsKwargs
    )

    return fig


def probplot(
    x: pd.DataFrame,
    mask: Optional[csr_matrix] = None,
    stDist: Callable = st.norm,
    distParams: Optional[list] = None,
    xlabel: str = 'Theoretical quantiles',
    ylabel: str = 'Ordered values',
    multiPlotsKwargs: dict = {}
) -> plt.figure:
    """
    Generate a probability plot.

    :param x: Input DataFrame.
    :param mask: Optional masking matrix.
    :param stDist: Statistical distribution function. Defaults to st.norm.
    :param distParams: Parameters for the distribution. Defaults to None.
    :param xlabel: Label for x-axis. Defaults to 'Theoretical quantiles'.
    :param ylabel: Label for y-axis. Defaults to 'Ordered values'.
    :param multiPlotsKwargs: Additional keyword arguments for multiple plots.
    :return: Matplotlib figure.
    """

    nplots = len(x.columns)
    if distParams == None: distParams = [None] * nplots

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        label = x.columns[i]
        if len(dataX) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes, None

        st.probplot(dataX, sparams=distParams[i], dist=stDist, plot=axes)

        axes.set_title(label)
        colorbarData = None
        return axes, colorbarData

    fig = multiplePlots(
        nplots,
        func,
        figTitle=fr'Probability plot [{stDist.name.capitalize()} distribution vs sample]',
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=None,
        **multiPlotsKwargs
    )
    return fig

# def qqplots(
#     dist1: List[np.ndarray],
#     dist2: List[np.ndarray],
#     labels: List[str],
#     quantile_range: List[float] = [0, 1],
#     num_quantiles: int = 100,
#     dist1_name: str = "Dist 1",
#     dist2_name: str = "Dist 2",
# ):
#     nlabels = len(labels)

#     quantiles = np.linspace(quantile_range[0], quantile_range[1], num=num_quantiles)
#     quantiles_dist1 = [np.array([np.quantile(dist1[i], q=quantiles[j]) for j in range(quantiles.shape[0])]) for i in range(nlabels)]
#     quantiles_dist2 = [np.array([np.quantile(dist2[i], q=quantiles[j]) for j in range(quantiles.shape[0])]) for i in range(nlabels)]

#     scatterplot(
#         quantiles_dist1, 
#         quantiles_dist2, 
#         labels=labels,
#         marker='.', correlationAxis='fit', cmap=None, xlabel=dist1_name, ylabel=dist2_name)
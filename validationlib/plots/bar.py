'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional
from packaging.version import Version

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse import csr_matrix
from joblib import delayed, Parallel

from .common import multiplePlots, dist_similarity
from ..misc.subsampling import maskApply

def coloredBarplot(
    labels: List[str],
    values: List[np.ndarray],
    ylabel: str = "",
    pvalueline: Optional[float] = None,
    colored_legend: bool = True,
):
    """
    Colored barplot.

    :param labels: List of labels.
    :param values: List of value arrays.
    :param ylabel: Label for y-axis.
    :param pvalueline: Column name of desired type of error in the errordf dataframe.
    :param colored_legend: Adds legend.
    :return: None.
    """
    length = len(labels)
    colors = ["C" + str(i) for i in range(length)]

    plotwidth = int(length + 3)
    plt.figure(figsize=(plotwidth, 8))
    plt.bar(labels, values, color=colors)

    if pvalueline != None:
        axes = plt.gca()
        xmin, xmax = axes.get_xlim()
        plt.plot(
            (xmin, xmax),
            (pvalueline, pvalueline),
            color="red",
            label="pvalue=" + str(pvalueline),
        )
    plt.ylabel(ylabel)

    # Custom legend. Instead of naming the bar on the x axis, we build a legend with a color code to support it
    if colored_legend == True:
        mpatch = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(length)]
        plt.legend(handles=mpatch, loc="upper right")
    return

def barplotFitSample(
    table: pd.DataFrame,
    test: str,
    metricName: str,
    alpha: float = 0.05,
    colorPair: list = ['green', 'red'],
    logscale: bool = False,
    multiPlotsKwargs: dict={}
) -> plt.figure:
    """
    Bar plot representing the fitness of a sample.

    :param table: Input DataFrame.
    :param test: Test type.
    :param metricName: Name of the metric.
    :param alpha: Alpha value (default=0.05).
    :param colorPair: Pair of colors for representation (default=['green', 'red']).
    :param logscale: Whether to use log scale (default=False).
    :param multiPlotsKwargs: Additional keyword arguments for multiple plots.
    :return: Matplotlib figure.
    """

    nplots = len(table.index)

    def func(i, axes: plt.Axes):
        label = table.index[i]

        values = [max(pvalue, 0.0003) for pvalue in table.iloc[i,:]]
        axes.bar(table.columns, values)

        for j, val in enumerate(values):
            if val < alpha: colorBar = colorPair[1]
            else:  colorBar = colorPair[0]

            axes.get_children()[j].set_color(colorBar)  
    
        axes.set_ylim([0, 1])
        if logscale: 
            axes.set_yticks([0, 0.05, 0.1, 0.5, 1])
            axes.set_yscale('log')
            axes.set_ylim([0.001, alpha])


        axes.axhline(y=alpha, color='blue', linestyle='dashed')
        axes.set_yticks(list(axes.get_yticks()) + [alpha])

        axes.set_title(label)
        colorbarData = None
        return axes, colorbarData

    fig = multiplePlots(
        nplots,
        func,
        figTitle=f"{test} test of compliance({metricName} distribution)",
        xlabel='Distribution',
        ylabel='p-value',
        clabel=None,
        **multiPlotsKwargs
    )

    return fig



def barplotFitDist(
        x: pd.DataFrame, 
        y: pd.DataFrame,
        dists: List[Callable],
        mask: Optional[csr_matrix] = None,
        test: str='KS',
        nbins: int=100,
        tails: str='No',
        # No, Left, Right. 
        # In the first category, bins contain only points within its limits
        # In the second one, bins contain all points up until their upper limit
        # In the third one, bins contain all points from their lower limit upwards
        colorPair: list = ['green', 'red'],
        xlabel: str='Independent variable',
        ylabel: str='Fitted variable',
        multiPlotsKwargs: dict={}
    ) -> plt.figure:
    """
    Bar plot for fitted distribution.

    :param x: Input DataFrame for x-axis.
    :param y: Input DataFrame for y-axis.
    :param dists: List of distribution functions.
    :param mask: Optional masking matrix.
    :param test: Test type (default='KS').
    :param nbins: Number of bins (default=100).
    :param tails: Tail type ('No', 'Left', 'Right') (default='No').
    :param colorPair: Pair of colors for representation (default=['green', 'red']).
    :param xlabel: Label for x-axis (default='Independent variable').
    :param ylabel: Label for y-axis (default='Fitted variable').
    :param multiPlotsKwargs: Additional keyword arguments for multiple plots.
    :return: Matplotlib figure.
    """

    nplots = len(x.columns)
    ndists = len(dists)
    colors = [colorPair[0], colorPair[1], "C7"]
    colorlabels = ["p-value > 0.05", "p-value < 0.05", "not enough data"]

    # Get the name of each dist
    distnames = [None] * len(dists)
    for i, dist in enumerate(dists):
        distnames[i] = dist.name.capitalize()
    

    def inner_loop(i):
        dataX = maskApply(x, mask, i)
        dataY = maskApply(y, mask, i)
        
        barh_color = [[None] * ndists for _ in range(nbins)]

        bounds = [min(dataX), max(dataX)]
        bins = np.linspace(bounds[0], bounds[1], nbins+1)
        bins[-1] += 0.1
        binnedIndices = np.digitize(dataX, bins)

        for m in range(nbins):
            if tails=='Left':
                condition = (binnedIndices <= m+1)
            elif tails=='Right':
                condition = (binnedIndices >= m+1)
            else:
                condition = (binnedIndices == m+1)

            dataYbin = dataY[condition]

            if len(dataYbin) > 2:
                enough = True
            else:
                enough = False

            for n in range(ndists):
                color = colors[2]
                if enough:
                    # Fit using the parameters from the previous iteration as starting point (saves time)
                    params = dists[n].fit(dataYbin)
                    arg, loc, scale = params[:-2], params[-2], params[-1]

                    # Random sampling from the fitted distribution
                    fitted = dists[n].rvs(size=len(dataYbin), loc=loc, scale=scale, *arg)
                    # Similarity test between the original sample and the random sample
                    _, pvalue = dist_similarity(dataYbin, fitted, test=test, report=False)
                    
                    if pvalue > 0.05:
                        color = colors[0]
                    else:
                        color = colors[1]

                barh_color[m][n] = color
        return barh_color
    
    result = Parallel(n_jobs=-1)(delayed(inner_loop)(i) for i in range(nplots))

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        label = x.columns[i]
        if len(dataX) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes, None

        bounds = [min(dataX), max(dataX)]
        left = bounds[0]
        delta = (bounds[1] - bounds[0]) / nbins

        for m in range(nbins):
            for n in range(ndists):
                axes.barh(distnames[n], delta, left=left, color=result[i][m][n])

            left += delta

        axes.set_xlim(bounds[0], bounds[1])
        if i == 0:
            axes.legend(colorlabels, loc='best')
            leg = axes.get_legend()
            if Version(mpl.__version__) >= Version("3.7"):
                leg.legend_handles[0].set_color(colors[0])
                leg.legend_handles[1].set_color(colors[1])
                leg.legend_handles[2].set_color(colors[2])
            else:
                leg.legendHandles[0].set_color(colors[0])
                leg.legendHandles[1].set_color(colors[1])
                leg.legendHandles[2].set_color(colors[2])

        # remove spines
        axes.spines["top"].set_visible(False)
        axes.spines["bottom"].set_visible(False)

        # adjust limits and draw grid lines
        axes.set_ylim(-0.5, axes.get_yticks()[-1] + 0.5)
        axes.xaxis.grid(color="gray", linestyle="dashed")

        axes.set_title(label)
        colorbarData = None
        return axes, colorbarData

    fig = multiplePlots(
        nplots,
        func,
        figTitle=f"Bar plot [{xlabel} vs. {test} test on {ylabel}'s fit] ({tails} tail)",
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=None,
        **multiPlotsKwargs
    )
    return fig


def doubleBar(
        x1: pd.DataFrame,
        x2: pd.DataFrame,
        mask: Optional[csr_matrix] = None,
        density: bool = True,
        logscale: bool = False,
        x1label: str = "x1label",
        x2label: str = "x2label",
        xlabel: str = "xlabel",
        label_significant_figures: int = 4,
        multiPlotsKwargs: dict = {}
) -> plt.figure:
    """
    Double bar plot.

    :param x1: Input DataFrame for first bar plot.
    :param x2: Input DataFrame for second bar plot.
    :param mask: Optional masking matrix.
    :param density: Whether to display density (default=True).
    :param logscale: Whether to use log scale (default=False).
    :param x1label: Label for first bar plot (default="x1label").
    :param x2label: Label for second bar plot (default="x2label").
    :param xlabel: Label for x-axis (default="xlabel").
    :param significant_figures: Number of significant figures for numeric labels (default=4).
    :param multiPlotsKwargs: Additional keyword arguments for multiple plots.
    :return: Matplotlib figure.
    """
    
    nplots = len(x1.columns)
    ylabel = 'Density' if density else 'Frequency'

    def func(i, axes: plt.Axes):
        dataX1 = maskApply(x1, mask, i, to_numpy=False)
        height1 = dataX1.value_counts()
        if density: height1 /= dataX1.shape[0]

        dataX2 = maskApply(x2, mask, i, to_numpy=False)
        height2 = dataX2.value_counts()
        if density: height2 /= dataX2.shape[0]

        label = x1.columns[i]
        categories = np.unique(np.concatenate([dataX1.unique(), dataX2.unique()]))
        x = np.arange(categories.shape[0])

        cat_labels = [f"{cat:.{max(label_significant_figures,np.ceil(np.log10(cat)).astype(int))}g}" if isinstance(cat, np.number) and not isinstance(cat, int) else cat for cat in categories]

        # Add missing categories to heights series
        for cat in categories:
            if cat not in height1.index: height1[cat] = 0
            if cat not in height2.index: height2[cat] = 0

        axes.bar(
            x - 0.2,
            height=height1[categories],
            width=0.4,
            label=x1label
        )

        axes.bar(
            x + 0.2,
            height=height2[categories],
            width=0.4,
            label=x2label
        )

        axes.set_xticks(x, cat_labels)

        if i==0: axes.legend()
        if logscale: axes.set_yscale("log")

        axes.set_title(label)
        colorbarData = None
        return axes, colorbarData
    
    fig = multiplePlots(
        nplots,
        func,
        figTitle=f'Double bar plot [{x1label} vs {x2label}]',
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=None,
        **multiPlotsKwargs
    )

    return fig

'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from ..misc.subsampling import maskApply
from .common import multiplePlots, dist_similarity

def lineplotFitPairs(
    x1: pd.DataFrame, 
    x2: pd.DataFrame, 
    mask: Optional[csr_matrix] = None,
    nbins=100,
    tails: str='No',
    test: str = 'KS',
    xlabel: str = 'xlabel',
    x1label: str = 'x1label',
    x2label: str = 'x2label',
    colorPair: list = ['green', 'red'],
    multiPlotsKwargs: dict = {}
) -> plt.figure:
    """
    Creates line plots for pairs of DataFrames x1 and x2, evaluating their similarity via statistical test.

    :param x1: First DataFrame for x-axis values.
    :param x2: Second DataFrame for x-axis values.
    :param mask: Mask for filtering.
    :param nbins: Number of bins (default=100).
    :param tails: Type of tails ('No', 'Left', 'Right') for p-value calculation (default='No').
    :param test: Statistical test ('KS', 'AD', etc.) for comparing distributions (default='KS').
    :param xlabel: Label for x-axis (default='xlabel').
    :param x1label: Label for x1-axis (default='x1label').
    :param x2label: Label for x2-axis (default='x2label').
    :param colorPair: Color pair for visualizing p-values (default=['green', 'red']).
    :param multiPlotsKwargs: Additional keyword arguments for multiplePlots (default={}).
    :return: Matplotlib figure.
    """

    nplots = len(x1.columns)

    def func(i, axes: plt.Axes):
        dataX1 = maskApply(x1, mask, i)
        dataX2 = maskApply(x2, mask, i)

        label = x1.columns[i]
        if len(dataX1) < 2 or len(dataX2) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes, None

        bounds = [min(dataX1), max(dataX1)]
        bins = np.linspace(bounds[0], bounds[1], nbins+1)
        bins[-1] += 0.1
        binnedIndices = np.digitize(dataX1, bins)

        pvalue = np.zeros(nbins)
        for m in range(nbins):
            if tails=='Left':
                condition = (binnedIndices <= m+1)
            elif tails=='Right':
                condition = (binnedIndices >= m+1)
            else:
                condition = (binnedIndices == m+1)
            if sum(condition) == 0: continue

            dataX1bin = dataX1[condition]
            dataX2bin = dataX2[condition]
            _, pvalue[m] = dist_similarity(dataX1bin, dataX2bin, test=test)

        xPlot = np.repeat(bins, 2)[1:-1]
        yPlot = np.repeat(pvalue, 2)
        
        axes.plot(xPlot, yPlot)
        axes.plot(bounds, [0.05, 0.05], color="black", label="0.05 threshold")

        axes.axhspan(0.05, 1, facecolor=colorPair[0], alpha=0.4)
        axes.axhspan(0, 0.05, facecolor=colorPair[1], alpha=0.4)

        axes.set_title(label)
        colorbarData = None
        return axes, colorbarData

    fig = multiplePlots(
        nplots,
        func,
        figTitle=fr'Line plot [{xlabel}-{x1label} vs {test} test on  {xlabel}-{x2label} similarity] ({tails} tail)',
        xlabel=xlabel,
        ylabel=f'p-value',
        clabel=None,
        **multiPlotsKwargs
    )
    return fig

def lineplots(
    x: pd.DataFrame,
    ylist: List[pd.DataFrame],
    mask: Optional[csr_matrix] = None,
    line_label: Union[List[str], str] = None,
    xlabel: str = "",
    ylabel: str = "",
    xlogscale=False,
    ylogscale=False,
    figHsize: float = 16,
    figAspectRatio: float = 4,
    nWorkers: int = 1
) -> plt.figure:
    """
    Creates line plots for multiple columns in DataFrame x against multiple DataFrames in ylist.

    :param x: DataFrame for x-axis values.
    :param ylist: List of DataFrames for y-axis values.
    :param mask: Mask for filtering.
    :param line_label: Label for each line (default=None).
    :param xlabel: Label for x-axis (default="").
    :param ylabel: Label for y-axis (default="").
    :param xlogscale: If True, x-axis uses log scale (default=False).
    :param ylogscale: If True, y-axis uses log scale (default=False).
    :param figHsize: Figure horizontal size (default=16).
    :param figAspectRatio: Figure aspect ratio (default=4).
    :param nWorkers: Number of workers for parallel processing (default=1).
    :return: Matplotlib figure.
    """
    
    nPlots = len(x.columns)

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        dataY = []
        for line in ylist:
            dataY.append(maskApply(line, mask, i))
        labelx = x.columns[i]
        if len(dataX) < 2:
            axes.set_title(labelx + " -NOT enough data in range-")
            return

        for j, line in enumerate(dataY):
            labely = "" if line_label is None else line_label[j]
            axes.plot(dataX, line, label=labely)

        if line_label is not None: axes.legend()

        if xlogscale: axes.set_xscale("log")
        if ylogscale: axes.set_yscale("log")

        axes.set_title(labelx)

        return axes, None

    fig = multiplePlots(
        nPlots,
        func,
        nWorkers,
        figHsize=figHsize,
        figAspectRatio=figAspectRatio,
        xlabel=xlabel,
        ylabel=ylabel,
        oneColorbar=False
    )
    return fig
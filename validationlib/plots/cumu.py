'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.sparse import csr_matrix

from ..misc.subsampling import maskApply
from .common import multiplePlots, histnbins

def doublecumulative(
    dist1: pd.DataFrame,
    dist2: pd.DataFrame,    
    nbins: int = 10,
    mask: csr_matrix = None,
    xlabel: str = "",
    label1: str = "",
    label2: str = "",
    cmap: Callable = plt.cm.jet,
    multiPlotsKwargs: dict = {}
):
    """
    WIP

    :param dist1: First DataFrame.
    :param dist2: Second DataFrame.
    :param mask: Mask for filtering.
    :param xlabel: Label for x-axis (default="").
    :param label1: Label for dist1 (default="").
    :param label2: Label for dist2 (default="").
    :param nbins: Number of bins (default=None).
    :param cmap: Color map (default=plt.cm.jet).
    :param multiPlotsKwargs: Additional keyword arguments for multiplePlots (default={}).
    :return: Matplotlib figure.
    """
    nplots = len(dist1.columns)
    colors = ["C0", "C1"]
    lab = [label1, label2]

    ylabel="Cumulative density"

    def func(i, axes: plt.Axes):
        data1 = maskApply(dist1, mask, i)
        data2 = maskApply(dist2, mask, i)
        label = dist1.columns[i]
        if len(data1) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes, None

        acu = [None] * 2
        binning = np.linspace(
            min([min(data1), min(data2)]),
            max([max(data1), max(data2)]),
            nbins,
        )

        axes.set_ylim([0,1])
        for index, dist in enumerate([data1, data2]):
            acu[index], _ = np.histogram(dist, binning)
            acu[index] = np.cumsum(acu[index]) / np.sum(acu[index])
            axes.plot(binning[:-1], acu[index], label=lab[index], c=colors[index], alpha=0.8, linewidth=2)
        for edge in range(nbins - 1):
            if acu[0][edge] > acu[1][edge]:
                axes.axvspan(binning[edge], binning[edge + 1], ymax=acu[0][edge], facecolor=colors[0], alpha=0.5)
            elif acu[0][edge] < acu[1][edge]:
                axes.axvspan(binning[edge], binning[edge + 1], ymax=acu[1][edge], facecolor=colors[1], alpha=0.5)

        axes.legend()
        axes.set_title(dist1.columns[i])
        colorbarData = None
        return axes, colorbarData
    
    fig = multiplePlots(
        nplots,
        func,
        figTitle=fr'Cumulative plot of [{xlabel}]',
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=None,
        **multiPlotsKwargs
    )
    return fig

def cumulative(
    x: pd.DataFrame,
    mask: csr_matrix = None,
    quantiles: List[float] = [0.50, 0.90, 0.95, 0.99],
    xlabel: str = "",
    bins: Optional[Union[str, int, list]] = None,
    cmap: Callable = plt.cm.jet,
    multiPlotsKwargs: dict = {}
) -> plt.figure:
    """
    Cumulative density plot.

    :param x: DataFrame for the plot.
    :param mask: Mask for filtering.
    :param xlabel: Label for x-axis (default="").
    :param bins: Bin definition (default=None). This accepts the same values as the bins parameter in numpy.histogram: int (number of bins), list (bin edges), or str (automatic binning method).
    :param cmap: Color map (default=plt.cm.jet).
    :param multiPlotsKwargs: Additional keyword arguments for multiplePlots (default={}).
    :return: Matplotlib figure.
    """

    quantiles = np.sort(quantiles)
    colors = cmap(np.linspace(0,1,len(quantiles)+1))
    q_heights = np.linspace(0.7,0.1,len(quantiles))
    # thresholds = [0.50,0.90,0.95,0.99]
    # heights = [0.7,0.5,0.3,0.1]


    nplots = len(x.columns)
    ylabel = 'Cumulative density'

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        label = x.columns[i]
        if len(dataX) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return

        # binning = np.linspace(min(dataX),max(dataX),nbin)

        acu, binning = np.histogram(dataX, bins)
        acu = np.cumsum(acu) / np.sum(acu)

        nbins = len(binning)-1

        axes.set_ylim([0,1])
        margin = 0.05*(max(dataX)-min(dataX))
        axes.set_xlim([min(dataX)-margin,max(dataX)+margin])
    
        axes.plot(binning, np.insert(acu, 0, 0), c='darkblue', alpha=0.8, linewidth=2)

        transform = axes.transData + axes.transAxes.inverted()

        for j, value in enumerate(quantiles):
            percent = np.quantile(dataX, q=value)
            axes.axvline(percent, ymax=transform.transform([0,q_heights[j]])[1], color="black", alpha=0.3)
            axes.axhline(q_heights[j], xmin=transform.transform([percent,0])[0], color="black", alpha=0.3)
            axes.text(transform.inverted().transform([1.08,0])[0], q_heights[j]+0.01, f'{value*100} %', fontsize=9, color='black', horizontalalignment='right', alpha=0.6)

        k = 0
        for j in range(nbins-1):
            if k<len(quantiles):
                if binning[j] >= np.quantile(dataX, q=quantiles[k]):
                    k += 1
            axes.axvspan(binning[j], binning[j+1], ymax=acu[j], facecolor=colors[k], alpha=0.8)
        
        axes.set_title(label)
        colorbarData = None
        return axes, colorbarData

    fig = multiplePlots(
        nplots,
        func,
        figTitle=fr'Cumulative plot of [{xlabel}]',
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=None,
        **multiPlotsKwargs
    )
    return fig
'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
"""
   Hist is a module for histogram plots. 
"""
from typing import Callable, Optional, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as st
from scipy.sparse import csr_matrix

from .common import multiplePlots, histnbins
from ..misc.subsampling import maskApply

def hextograms(
    x: pd.DataFrame,
    y: pd.DataFrame,
    mask: csr_matrix = None,
    grid: int = 15,
    logscale = False,
    figHsize: float = 16,
    figAspectRatio: float = 4,
    cmap: Callable = plt.cm.Blues,
    oneColorbar: bool = False,
    xlabel: str = None,
    ylabel: str = None,
    clabel: str = 'Density',
    nWorkers: int = 1
) -> plt.figure:
    """
    Creates hexbin plots for multiple columns in x and y DataFrames.

    :param x: DataFrame for x-axis values.
    :param y: DataFrame for y-axis values.
    :param mask: Mask for filtering.
    :param grid: Hexbin grid size (default=15).
    :param logscale: If True, bins are on log scale (default=False).
    :param figHsize: Figure horizontal size (default=16).
    :param figAspectRatio: Figure aspect ratio (default=4).
    :param cmap: Color map (default=plt.cm.Blues).
    :param oneColorbar: If True, uses one color bar for all plots (default=False).
    :param xlabel: Label for x-axis (default=None).
    :param ylabel: Label for y-axis (default=None).
    :param clabel: Colorbar label (default='Density').
    :param nWorkers: Number of workers for parallel processing (default=1).
    :return: Matplotlib figure.
    """

    nPlots = len(x.columns)
    if logscale:
        logscale = "log"
    else:
        logscale = None

    def func(i, axes: plt.Axes):

        datax = maskApply(x, mask, i)
        datay = maskApply(y, mask, i)

        if len(datax) < 2:
            axes.set_title(x.columns[i] + " -NOT enough data in range-")
            return axes

        colorbarData = axes.hexbin(datax, datay, gridsize=grid, bins=logscale, cmap=cmap)
        axes.set_title(x.columns[i])

        return axes, colorbarData

    fig = multiplePlots(
        nPlots,
        func,
        nWorkers,
        figHsize=figHsize,
        figAspectRatio=figAspectRatio,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=clabel,
        oneColorbar=oneColorbar
    )
    return fig

def hist2D(
    x: pd.DataFrame,
    y: pd.DataFrame,
    mask: Optional[csr_matrix] = None,
    bins: List[int] = None,
    logscale: bool = False,
    figHsize: float = 16,
    figAspectRatio: float = 4,
    correlationAxis: Optional[str] = None,
    fit_info: str = "default",
    significant_figures: int = 3,
    cmap: Callable = plt.cm.jet,
    oneColorbar: bool = False,
    xlabel: str = None,
    ylabel: str = None,
    scale: str = 'density',
    nWorkers: int = 1
) -> plt.figure:
    """
    Creates 2D histograms for multiple columns in x and y DataFrames.

    :param x: DataFrame for x-axis values.
    :param y: DataFrame for y-axis values.
    :param mask: Mask for filtering.
    :param bins: Number of bins along x and y axes (default=[10, 10]).
    :param scale: Scale for histogram (default="density"). Accepted values are:
        - "density": Normalizes the histogram to an empirical probability density distribution (so that the integral equals to one).
        - "frequency": Shows the frequency of each bin.
        - "proportion": Shows the frequency of each bin normalized to the total number of data points.
    :param logscale: If True, uses log scale (default=False).
    :param figHsize: Figure horizontal size (default=16).
    :param figAspectRatio: Figure aspect ratio (default=4).
    :param cmap: Color map (default=plt.cm.jet).
    :param oneColorbar: If True, uses one color bar for all plots (default=False).
    :param xlabel: Label for x-axis (default=None).
    :param ylabel: Label for y-axis (default=None).
    :param nWorkers: Number of workers for parallel processing (default=1).
    :return: Matplotlib figure.
    """
    if bins is None:
        bins = [10, 10]

    nPlots = len(x.columns)

    if logscale:
        norm = mcolors.LogNorm()
    else:
        norm = None

    if scale == 'density':
        clabel = 'Density'
    elif scale == 'frequency':
        clabel = 'Frequency'
    elif scale == 'proportion':
        clabel = 'Proportion'
    else:
        raise ValueError(f"Unknown scale: {scale}. Accepted values are 'density', 'frequency', and 'proportion'.")

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        dataY = maskApply(y, mask, i)
        label = x.columns[i]
        if len(dataX) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes

        if scale == 'density':
            density = True
            weights = None
        elif scale == 'frequency':
            density = False
            weights = None
        elif scale == 'proportion':
            density = False
            weights = np.ones_like(dataX) / len(dataX)
        else:
            raise ValueError(f"Unknown scale: {scale}. Accepted values are 'density', 'frequency', and 'proportion'.")

        a, b = np.min(dataX), np.max(dataX)
        x_corraxs = np.linspace(a, b, 1000)
        if correlationAxis == "+":
            y_corraxs = x_corraxs
            axes.plot(x_corraxs, y_corraxs, color="black", linewidth=0.5, label="y=x")
            axes.legend(loc="best")
        elif correlationAxis == "-":
            y_corraxs = np.flip(x_corraxs)
            axes.plot(x_corraxs, y_corraxs, color="black", linewidth=0.5, label="y=-x")
            axes.legend(loc="best")
        elif correlationAxis == "fit":
            m, bb = np.polyfit(dataX, dataY, deg=1)
            yA, yB = m * a + bb, m * b + bb
            mNorm = m * (b - a) / (np.max(dataY) - np.min(dataY))
            y_corraxs = m * x_corraxs + bb
            r, _ = st.pearsonr(dataX, dataY)
            if fit_info == "normalized":
                axes.plot(x_corraxs, y_corraxs, label=f"Normalized slope: {mNorm:.{significant_figures}g}\nR2: {r:.{significant_figures}g}", color="black", linewidth=0.5)
            elif fit_info == "default":
                axes.plot(x_corraxs, y_corraxs, label=f"y={m:.{significant_figures}g}x + {bb:.{significant_figures}g}\nR2: {r:.{significant_figures}g}", color="black", linewidth=0.5)
            else:
                raise ValueError("fit_info must be 'default' or 'normalized'")
            axes.legend(loc="best")

        _, _, _, colorbarData = axes.hist2d(dataX, dataY,
                                            bins=bins, norm=norm,
                                            cmap=cmap, weights=weights,
                                            density=density,)
        axes.set_title(label)

        return axes, colorbarData

    fig = multiplePlots(
        nPlots,
        func,
        nWorkers,
        figHsize=figHsize,
        figAspectRatio=figAspectRatio,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=clabel,
        oneColorbar=oneColorbar
    )
    return fig

def histogram(
    x: pd.DataFrame,
    mask: Optional[csr_matrix] = None,
    nbins: Optional[int] = None,
    scale: str = "density",
    logscale: bool = False,
    kde: bool = False,
    trimStds: Optional[float] = None,
    xlabel: str = None,
    multiPlotsKwargs: dict = None
) -> plt.figure:
    """
    Creates histograms for multiple columns in DataFrame x.

    :param x: DataFrame for plotting histograms.
    :param mask: Mask for filtering.
    :param nbins: Number of bins (default=None).
    :param scale: Scale for histogram (default="density"). Accepted values are:
        - "density": Normalizes the histogram to an empirical probability density distribution (so that the integral equals to one).
        - "frequency": Shows the frequency of each bin.
        - "proportion": Shows the frequency of each bin normalized to the total number of data points.
    :param logscale: If True, y-axis is logarithmic (default=False).
    :param kde: If True, displays kernel density estimation (default=False).
    :param trimStds: Trims data within specified standard deviations (default=None).
    :param xlabel: Label for x-axis (default=None).
    :param multiPlotsKwargs: Additional keyword arguments for multiplePlots (default={}).
    :return: Matplotlib figure.
    """
    if multiPlotsKwargs is None:
        multiPlotsKwargs = {}
    nplots = len(x.columns)
    ylabel = scale.capitalize()

    trimMsg = '' if trimStds is None else fr'{"Data" if xlabel is None else ""} inside $\mu \pm {trimStds}\sigma$'
    title_str = (f'Histogram of [{xlabel}] ' if xlabel is not None else '') + trimMsg

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        label = x.columns[i]
        if len(dataX) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes

        nbin = histnbins(len(dataX)) if nbins is None else nbins

        if trimStds is not None:
            mean, stds = np.mean(dataX), np.std(dataX)
            dataX = np.clip(dataX, mean-trimStds*stds, mean+trimStds*stds)

        if scale == "density":
            density = True
            weights = None
        elif scale == "frequency":
            density = False
            weights = None
        elif scale == "proportion":
            density = False
            weights = np.ones_like(dataX) / len(dataX)
        else:
            raise ValueError(f"Unknown scale: {scale}. Accepted values are 'density', 'frequency', and 'proportion'.")

        axes.hist(dataX, bins=nbin, density=density, weights=weights, alpha=0.6, histtype="bar", ec="white")

        if kde:
            ymin, _ = axes.get_ylim()
            kernel = st.gaussian_kde(dataX)
            xplot = np.linspace(min(dataX), max(dataX), 100)
            axes.plot(xplot, np.maximum(kernel(xplot), ymin),
                      color="C0", linewidth=2)
            axes.set_ylim(ymin)

        if logscale:
            axes.set_yscale("log")
        axes.set_title(label)
        colorbarData = None
        return axes, colorbarData

    fig = multiplePlots(
        nplots,
        func,
        figTitle=title_str,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=None,
        **multiPlotsKwargs
    )
    return fig

def doubleHistogram(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    mask: Optional[csr_matrix] = None,
    scale: str = "density",
    bins: Optional[Union[str, int, list]] = None,
    logscale: bool = False,
    trimStds: Optional[float] = None,
    x1label: str = "x1label",
    x2label: str = "x2label",
    xlabel: str = None,
    multiPlotsKwargs: dict = None
) -> plt.figure:
    """
    Creates histograms for comparing two sets of multiple columns in DataFrames x1 and x2.

    :param x1: First DataFrame for plotting histograms.
    :param x2: Second DataFrame for plotting histograms.
    :param mask: Mask for filtering.
    :param scale: Scale for histogram (default="density"). Accepted values are:
        - "density": Normalizes the histogram to an empirical probability density distribution (so that the integral equals to one).
        - "frequency": Shows the frequency of each bin.
        - "proportion": Shows the frequency of each bin normalized to the total number of data points.
    :param bins: Union[str, int, list], optional
        Specification of the bins. Accepted values are:
        - String: Automatic binning methods accepted by numpy.histogram_bin_edges (see numpy docs).
        - Integer: Number of bins to use.
        - List: List of bin edges.
    :param logscale: If True, y-axis is logarithmic (default=False).
    :param trimStds: Trims data within specified standard deviations (default=None).
    :param x1label: Label for x1-axis (default="x1label").
    :param x2label: Label for x2-axis (default="x2label").
    :param xlabel: Label for x-axis (default="xlabel").
    :param multiPlotsKwargs: Additional keyword arguments for multiplePlots (default={}).
    :return: Matplotlib figure.
    """
    if multiPlotsKwargs is None:
        multiPlotsKwargs = {}

    nplots = len(x1.columns)
    ylabel = scale.capitalize()
    trimMsg = '' if trimStds is None else fr'inside $\mu \pm {trimStds}\sigma$'

    if bins is None:
        bins = "sturges"

    def func(i, axes: plt.Axes):
        dataX1 = maskApply(x1, mask, i)
        dataX2 = maskApply(x2, mask, i)
        label = x1.columns[i]
        if len(dataX1) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes

        # nbin = histnbins(len(dataX1)) if nbins is None else nbins

        if trimStds is not None:
            mean, stds = np.mean(dataX1), np.std(dataX1)
            dataX1 = np.clip(dataX1, mean-trimStds*stds, mean+trimStds*stds)
            mean, stds = np.mean(dataX2), np.std(dataX2)
            dataX2 = np.clip(dataX2, mean-trimStds*stds, mean+trimStds*stds)

        if scale == "density":
            density = True
            weights = None
        elif scale == "frequency":
            density = False
            weights = None
        elif scale == "proportion":
            density = False
            weights = [np.ones_like(dataX1) / len(dataX1), np.ones_like(dataX2) / len(dataX2)]
        else:
            raise ValueError(f"Unknown scale: {scale}. Accepted values are 'density', 'frequency', and 'proportion'.")

        axes.hist(
            [dataX1, dataX2],
            bins=bins,
            density=density,
            weights=weights,
            label=[x1label, x2label],
        )
        if i==0:
            axes.legend()
        if logscale:
            axes.set_yscale("log")

        axes.set_title(label)
        colorbarData = None
        return axes, colorbarData

    fig = multiplePlots(
        nplots,
        func,
        figTitle=f'Double histogram [{x1label} vs {x2label}] {trimMsg}',
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=None,
        **multiPlotsKwargs
    )
    return fig

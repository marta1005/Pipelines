'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import scipy.stats as st
from scipy.sparse import csr_matrix

from .common import multiplePlots
from ..misc.subsampling import maskApply
from ..misc.utils import check_column_number

def binnedScatterplot(
    x: pd.DataFrame,
    y: pd.DataFrame,
    mask: csr_matrix = None,
    bins: int = None,
    elems: int = None,
    categorical: bool = False,
    x_jitter: float = 0,
    show_mean:bool = True,
    show_percentile:List[float] = None,
    logscale=False,
    show_bins=False,
    xlabel: str = "",
    ylabel: str = "",
    figHsize: float = 16,
    figAspectRatio: float = 4,
    nWorkers: int = 1
):
    """
    Generates a binned scatterplot.

    Parameters:
    x : pd.DataFrame
        Dataframe for x-axis.
    y : pd.DataFrame
        Dataframe for y-axis.
    mask : csr_matrix
        Masking matrix.
    bins : int, optional
        Number of bins, by default None.
    elems : int, optional
        Elements per bin, by default None.
    categorical : bool, optional
        Categorical plotting, by default False.
    x_jitter : float, optional
        Jitter for x-axis, by default 0.
    show_mean : bool, optional
        Display mean, by default True.
    show_percentile : List[float], optional
        Display percentiles, by default None.
    logscale : bool, optional
        Log scale, by default False.
    show_bins : bool, optional
        Display bins, by default False.
    xlabel : str, optional
        X-axis label, by default "".
    ylabel : str, optional
        Y-axis label, by default "".
    figHsize : float, optional
        Figure horizontal size, by default 16.
    figAspectRatio : float, optional
        Figure aspect ratio, by default 4.
    nWorkers : int, optional
        Number of workers, by default 1.

    Returns:
    plt.figure
        Matplotlib figure.
    """

    nPlots = len(x.columns)

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        dataY = maskApply(y, mask, i)
        label = x.columns[i]
        if len(dataX) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes, None

        axes.set_prop_cycle(color=list(mcolors.TABLEAU_COLORS)[1:])

        if logscale:
            dataX = dataX[dataX > 0]
            dataY = dataY[dataY > 0]

        if bins is not None:
            if logscale:
                bins_ = np.logspace(start=np.log10(dataX[0]),
                                    stop=np.log10(dataX[-1]), base=10, num=bins)
            else:
                bins_ = bins

            _, binned_x = np.histogram(dataX, bins=bins_)
            binned_x[0] = dataX[0]
            binned_x[-1] = dataX[-1]

        elif elems is not None:
            bin_indices = np.arange(0, dataX.shape[0], elems)
            if dataX[bin_indices[-1]] < dataX[-1]:
                binned_x = np.empty(bin_indices.shape[0] + 1)
                binned_x[0 : bin_indices.shape[0]] = dataX[bin_indices]
                binned_x[-1] = dataX[-1]
            else:
                binned_x = dataX[bin_indices]

        elif categorical:
            pass

        else:
            raise ValueError("At least one of the following parameters must be set:"
                             " bins, elems, categorical.")

        if categorical:
            x_labels = dataX
            plotted_x = list(range(len(dataX)))
            binned_mean_x = np.array(plotted_x)
            binned_y = np.mean(dataY, axis=1)

            if show_percentile:
                binned_percentile = [None]*len(show_percentile)
                for k, p in enumerate(show_percentile):
                    binned_percentile[k] = np.percentile(dataY, q=p, axis=1)

            plotted_y = dataY.T.flatten()
            plotted_x = np.array(plotted_x*int(plotted_y.shape[0]/len(plotted_x)))

        else:
            binned_y = np.empty(binned_x.shape[0] - 1)
            binned_mean_x = np.empty(binned_x.shape[0] - 1)
            if show_percentile:
                binned_percentile = [np.empty(binned_x.shape[0] - 1)]*len(show_percentile)
            for j in range(binned_y.shape[0]):
                x_bin = dataX[(dataX >= binned_x[j]) & (dataX <= binned_x[j + 1])]
                binned_mean_x[j] = np.mean(x_bin) if len(x_bin) > 0 else (binned_x[j] + binned_x[j + 1]) / 2

                y_bin = dataY[(dataX >= binned_x[j]) & (dataX <= binned_x[j + 1])]
                binned_y[j] = np.mean(y_bin) if len(y_bin) > 0 else binned_y[j - 1]

                if show_percentile:
                    for k, p in enumerate(show_percentile):
                        binned_percentile[k][j] = np.percentile(y_bin, q=p) if len(y_bin) > 0 else binned_percentile[k][j - 1]

            if show_bins:
                bin_colors = [(0.95, 0.95, 0.95), (0.98, 0.98, 0.98)]
                for j in range(binned_x.shape[0] - 1):
                    axes.axvspan(binned_x[j], binned_x[j + 1], facecolor=bin_colors[j % 2], alpha=1)

            plotted_x, plotted_y = dataX, dataY

        plotted_x = plotted_x + x_jitter*(min(np.diff(plotted_x)))*st.norm.rvs(size=plotted_x.size)

        if show_mean:
            axes.plot(binned_mean_x, binned_y, ".-", label="Mean")

        if show_percentile:
            for k, p in enumerate(show_percentile):
                axes.plot(binned_mean_x, binned_percentile[k], ".-", label="p={:.2f}".format(p))

        axes.scatter(plotted_x, plotted_y, s=1, c="tab:blue", alpha=0.2)

        if categorical:
            axes.set_xticks(binned_mean_x)
            axes.set_xticklabels(x_labels)

        if logscale:
            axes.set_xscale("log")
        axes.set_title(label)

        axes.legend()

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

def binnedAverage(
    x: pd.DataFrame,
    y: pd.DataFrame,
    mask: csr_matrix = None,
    method: str = "CMA",
    CMAnOrder: int = 12,
    DWAdFactor: float = 0.01,
    nbins: int = 100,
    scatter: bool = False,
    scatterSize: float = 0.1,
    scatterColors: Optional[pd.DataFrame] = None,
    scatterColorLimits: Optional[List[float]] = None,
    oneColorbar: bool = False,
    cmap: Callable = plt.cm.coolwarm_r,
    boundScatter: bool = False,
    figHsize: float = 16,
    figAspectRatio: float = 4,
    xlabel: str = "",
    ylabel: str = "",
    clabel: str = "",
    nWorkers: int = 1
) -> plt.figure:
    """
    Centered Moving Average
    https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc422.htm
    Distance Weighted Average
    https://encyclopediaofmath.org/wiki/Distance-weighted_mean

    Generates a binned average plot.

    :param x: pd.DataFrame
        Dataframe for x-axis.
    :param y: pd.DataFrame
        Dataframe for y-axis.
    :param mask: csr_matrix
        Masking matrix.
    :param method: str, optional
        Method for averaging, by default "CMA".
    :param CMAnOrder: int, optional
        CMA order, by default 12.
    :param DWAdFactor: float, optional
        DWA factor, by default 0.01.
    :param nbins: int, optional
        Number of bins, by default 100.
    :param scatter: bool, optional
        Scatter plot, by default False.
    :param scatterSize: float, optional
        Scatter size, by default 0.1.
    :param scatterColors: Optional[pd.DataFrame], optional
        Scatter colors, by default None.
    :param scatterColorLimits: Optional[List[float]], optional
        Scatter color limits, by default None.
    :param oneColorbar: bool, optional
        Single colorbar, by default False.
    :param cmap: Callable, optional
        Colormap, by default plt.cm.coolwarm_r.
    :param boundScatter: bool, optional
        Bound scatter, by default False.
    :param clabel: str, optional
        Colorbar label, by default "".
    :param figHsize: float, optional
        Figure horizontal size, by default 16.
    :param figAspectRatio: float, optional
        Figure aspect ratio, by default 4.
    :param xlabel: str, optional
        X-axis label, by default "".
    :param ylabel: str, optional
        Y-axis label, by default "".
    :param nWorkers: int, optional
        Number of workers, by default 1.

    :return: plt.figure
        Matplotlib figure.
    """
    nplots = len(x.columns)

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        dataY = maskApply(y, mask, i)
        label = x.columns[i]
        if len(dataX) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes, None

        ### Binning and calculation of mean + percentils
        xMin, xMax = np.min(dataX), np.max(dataX)
        yMean = np.mean(dataY)

        if method == "CMA":
            xPlot, yPlot = np.linspace(xMin, xMax, nbins), np.full((nbins,3),None)
            delta = xPlot[1]-xPlot[0]

            for j in range(nbins):
                if j<CMAnOrder:
                    subset = dataY[(dataX > xPlot[max(j-CMAnOrder,0)]) & (dataX < xPlot[j+CMAnOrder])]
                elif j>nbins-CMAnOrder-1:
                    subset = dataY[(dataX > xPlot[j-CMAnOrder]) & (dataX < xPlot[min(j+CMAnOrder,nbins-1)])]
                else:
                    subset = dataY[(dataX > xPlot[j-CMAnOrder]) & (dataX < xPlot[j+CMAnOrder])]

                if len(subset) == 0: continue

                yPlot[j,0] = np.percentile(subset, 5)
                yPlot[j,1] = np.mean(subset)
                yPlot[j,2] = np.percentile(subset, 95)

            xPlot = np.clip(np.sort(np.concatenate((xPlot-delta/2,xPlot+delta/2))), xMin, xMax)
            yPlot = np.repeat(yPlot, 2, axis=0)
            axes.plot(xPlot, yPlot[:,0], label=f'{method} percentile 5', color='r')
            axes.plot(xPlot, yPlot[:,1], label=f'{method} mean', color='black')
            axes.plot(xPlot, yPlot[:,2], label=f'{method} percentile 95', color='b')

        elif method == "DWA":
            xPlot, yPlot = np.linspace(xMin, xMax, nbins), np.full(nbins,None)
            delta = xPlot[1]-xPlot[0]

            for j in range(nbins):
                subset = np.absolute(dataX - xPlot[j])
                weights = np.divide(1, (np.clip(subset / max(subset),DWAdFactor,1))**0.5)
                yPlot[j] = np.average(dataY, weights=weights)

            xPlot = np.clip(np.sort(np.concatenate((xPlot-delta/2,xPlot+delta/2))), xMin, xMax)
            yPlot = np.repeat(yPlot, 2, axis=0)
            axes.plot(xPlot, yPlot, label=f'{method} mean', color='black')

        binColors = [(0.97, 0.97, 0.97), (1, 1, 1)]
        for j in range(nbins - 1):
            axes.axvspan(xPlot[2*j], xPlot[2*j + 2], facecolor=binColors[j % 2], alpha=1)

        ### Plotting mean and percentils
        axes.plot((xMin, xMax), (yMean, yMean), linestyle=":", label="Global Mean")
        lims = axes.get_ylim()

        ### Scatterplot on top
        if scatter:
            if scatterColors is not None:
                dataColors = maskApply(scatterColors, mask, i)
                if scatterColorLimits is not None:
                    colors = np.clip(dataColors,
                                     scatterColorLimits[0],
                                     scatterColorLimits[1])
                else: colors = dataColors.copy()
                colorbarData = axes.scatter(
                    dataX,
                    dataY,
                    c=colors,
                    s=scatterSize,
                    norm=mcolors.TwoSlopeNorm(vcenter=0),
                    cmap=cmap
                )
            else:
                axes.scatter(dataX, dataY, s=scatterSize)
                colorbarData = None
            if boundScatter:
                axes.set_ylim(lims)
        else: colorbarData = None

        axes.legend()
        axes.set_title(label)
        return axes, colorbarData

    fig = multiplePlots(
        nplots,
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

def scatterplot(
    x: pd.DataFrame,
    y: pd.DataFrame,
    mask: Optional[csr_matrix] = None,
    c: Optional[pd.DataFrame] = None,
    markerSize: float = 1,
    marker: Optional[str] = "+",
    trimStds: Optional[float] = None,
    correlationAxis: Optional[str] = None,
    fit_info: str = "default",
    significant_figures: int = 3,
    cmap: Callable = plt.cm.coolwarm_r,
    cmapCenter: float = 0,
    xlogscale: bool = False,
    ylogscale: bool = False,
    xlabel: str = "xlabel",
    ylabel: str = "ylabel",
    title_prefix: str = "",
    clabel: str = "clabel",
    multiPlotsKwargs: dict = None
) -> plt.figure:
    """
    Generates a scatterplot.

    :param x: pd.DataFrame
        Dataframe for x-axis.
    :param y: pd.DataFrame
        Dataframe for y-axis.
    :param mask: Optional[csr_matrix], optional
        Masking matrix, by default None.
    :param c: Optional[pd.DataFrame], optional
        Dataframe for color mapping, by default None.
    :param markerSize: float, optional
        Marker size, by default 1.
    :param marker: Optional[str], optional
        Marker style, by default "+".
    :param trimStds: Optional[float], optional
        Trim standard deviations for color mapping, by default None.
    :param correlationAxis: Optional[str], optional
        Correlation axis display, by default None.
    :param fit_info: str, optional
        Correlation information, by default "default".
            - "default": Display slope, intercept, and R2.
            - "normalized": Display normalized slope and R2.
    :param cmap: Callable, optional
        Colormap function, by default plt.cm.coolwarm_r.
    :param cmapCenter: float, optional
        Center point for colormap, by default 0.
    :param xlabel: str, optional
        X-axis label, by default "xlabel".
    :param ylabel: str, optional
        Y-axis label, by default "ylabel".
    :param title_prefix: str, optional
        Prefix for plot title, by default "".
    :param clabel: str, optional
        Colorbar label, by default "clabel".
    :param multiPlotsKwargs: dict, optional
        Additional keyword arguments for multiplePlots function, by default {}.

    :return: plt.figure
        Matplotlib figure.
    """

    if c is not None:
        x, y, _, [c] = check_column_number(x, y, y_like=[c])
    else:
        x, y, _, _ = check_column_number(x, y)

    if multiPlotsKwargs is None:
        multiPlotsKwargs = {}
    nplots = len(x.columns)
    trimMsg = fr'(color trimmed to $\mu \pm {trimStds}\sigma$)' if trimStds is not None and c is not None else '' 

    def func(i, axes: plt.Axes):
        dataX = maskApply(x, mask, i)
        dataY = maskApply(y, mask, i)
        label = x.columns[i]
        if len(dataX) < 2:
            axes.set_title(label + " -NOT enough data in range-")
            return axes, None

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

        if c is not None and cmap is not None:
            dataC = maskApply(c, mask, i)
            if trimStds is not None:
                mean, stds = np.mean(dataC), np.std(dataC)
                dataC = np.clip(dataC, mean-trimStds*stds, mean+trimStds*stds)
            colorbarData = axes.scatter(
                dataX,
                dataY,
                s=markerSize,
                marker=marker,
                c=dataC,
                norm=TwoSlopeNorm(vcenter=cmapCenter),
                cmap=cmap,
            )
        else:
            axes.scatter(dataX, dataY, s=markerSize, marker=marker, color="b")
            colorbarData = None
        axes.set_title(label)

        if xlogscale:
            axes.set_xscale("log")
        if ylogscale:
            axes.set_yscale("log")

        return axes, colorbarData

    figTitle = "" if title_prefix == "" else f"{title_prefix}\n"
    figTitle += f'Scatter plot [{xlabel} vs {ylabel}] {trimMsg}'

    fig = multiplePlots(
        nplots,
        func,
        figTitle=figTitle,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=clabel,
        **multiPlotsKwargs
    )
    return fig

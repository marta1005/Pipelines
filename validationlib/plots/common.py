'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from joblib import delayed, Parallel
import scipy.stats as st

# WIP
def dist_similarity(*distlist, test="KS", report=False):
    """
    2 sample similarity test.

    Unless it's specified otherwise, H0=Both samples have the same underlying distribution

    #### Inputs
    ###### dist1 : list, dist2 : list
        Distributions to compare
    ###### test: str
        AD(Ksample Anderson-Darling)- Checks for the distance between the cumulative of the equally sized samples

        KS(2sample Kolmogorov-Smirnov)- Assumes observations to be independent

        WL(2sample Wilcoxon)- X and Y are samples of matched independent pairs, the distribution of the residue is symmetrical  

        MW(2sample Mann-Whitney U)- Assumes X and Y to be continuous and mutually independent from each other  

        KW(Ksample Kruskal-Wallis)- Does not assume residuals following a normal distribution unlike ANOVA. Can be seen as MW extended to more than 2 samples.

    :param distlist: Distributions to compare.
    :param test: Test type (default='KS').
    :param report: If True, prints the output (default=False).
    :return: Statistic and p-value.
    """
    if test == "AD":
        test = "Ksample Anderson-Darling"
        # n = len(distlist[0])
        AD, _, pvalue = st.anderson_ksamp(distlist)
        # AD = AD * (1 + 0.75 / n + 2.25 / n ** 2) # Modified value of A2 statistic (see reference below, p123)
        statistic = AD
        # # Reference used to obtain the p-value: R.B. D'Augostino and M.A. Stephens, Eds., 1986, Goodness-of-Fit Techniques, Marcel Dekker. p127
        # if AD >= 0.6:
        #     pvalue = np.exp(1.2937 - 5.709 * AD + 0.0186 * (AD ** 2))
        # elif AD >= 0.34:
        #     pvalue = np.exp(0.9177 - 4.279 * AD - 1.38 * (AD ** 2))
        # elif AD > 0.2:
        #     pvalue = 1 - np.exp(-8.318 + 42.796 * AD - 59.938 * (AD ** 2))
        # else:
        #     pvalue = 1 - np.exp(-13.436 + 101.14 * AD - 223.73 * (AD ** 2))
    elif test == "KS":
        statistic, pvalue = st.ks_2samp(*distlist)
        test = "2sample Kolmogorov-Smirnov"
    elif test == "WL":
        statistic, pvalue = st.wilcoxon(*distlist)
        test = "2sample Wilcoxon"
    elif test == "MW":
        statistic, pvalue = st.mannwhitneyu(*distlist)
        test = "2sample Mann-Whitney U"
    elif test == "KW":
        statistic, pvalue = st.kruskal(*distlist)
        test = "Ksample Kruskal-Wallis"
    else:
        raise ValueError("Error: Choose a valid test")

    if report:
        print(str(test) + " test results: statistic=" + str(round(statistic, 3)) + ", pvalue=" + str(round(pvalue, 3)))
    return statistic, pvalue
# WIP

def histnbins(vectorLength: int) -> int:
    """
    Auxiliary internal function: Sturges rule of thumb to calculate number of bins in a histogram.
    """
    return int(1 + np.ceil(np.log2(vectorLength)))

#oneColorbar doesn't work when parallelizing the work. Fix
def multiplePlots(
    nplots: int,
    func: Callable,
    nWorkers: int = 1,
    figHsize: float = None,
    figAspectRatio: float = 4,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    clabel: Optional[str] = None,
    figTitle: Optional[str] = None,
    oneColorbar: Optional[bool] = False,
    colorbarKwargs: Optional[dict] = {},
    tight_layout: Optional[bool] = False,
    xticks_rotation: Optional[float] = None,
    func_kwargs: Optional[dict] = {}
) -> plt.figure:
    """
    Multiple plots.

    :param nplots: Total number of plots in figure.
    :param func: Function that makes the necessary graphs.
    :param nWorkers: Number of parallel workers (default=1).
    :param figHsize: Figure height size (default=None).
    :param figAspectRatio: Figure aspect ratio (default=4).
    :param xlabel: Label for x-axis (default=None).
    :param ylabel: Label for y-axis (default=None).
    :param clabel: Label for colorbar (default=None).
    :param figTitle: Title of the figure (default=None).
    :param oneColorbar: Whether to use one colorbar for all subplots (default=False).
    :param colorbarKwargs: Additional keyword arguments for colorbar (default={}).
    :param tight_layout: Whether to apply tight layout (default=False).
    :param xticks_rotation: Rotation angle for x-axis ticks (default=None).
    :param func_kwargs: Additional keyword arguments for the function.
    :return: Matplotlib figure.
    """

    if figHsize is None:
        if nplots > 4:
            figHsize = 18
        elif nplots == 1:
            figHsize = 8
            figAspectRatio = 2
        else:
            figHsize = 14

    if nplots > 4:
        cols = 3
    elif nplots==1:
        cols = 1
    else:
        cols = 2

    rows = int(np.ceil(nplots / cols))

    figSize = (figHsize, figHsize / figAspectRatio * rows)
    fig, ax = plt.subplots(rows, cols, figsize=figSize)
    if cols == 1 and rows == 1: ax = np.array(ax)

    if figTitle is not None:
        title = fig.suptitle(figTitle, fontsize=20)
        title.set_y(1)

    if nWorkers==1:
        for i, axes in enumerate(ax.flatten()):
            # Delete unused axes
            if i >= nplots:
                fig.delaxes(axes)
                continue

            # Compute
            axes, colorbarData = func(i, axes, **func_kwargs)

            if isinstance(xlabel, str):
                if i >= nplots - cols: axes.set_xlabel(xlabel)
                else: axes.set_xlabel('')
            elif hasattr(xlabel, '__iter__'):
                axes.set_xlabel(xlabel[i])

            if isinstance(ylabel, str):
                if i % cols == 0: axes.set_ylabel(ylabel)
                else: axes.set_ylabel('')
            elif hasattr(ylabel, '__iter__'):
                axes.set_ylabel(ylabel[i])

            # xticks rotation
            plt.xticks(rotation=xticks_rotation)

            # Colorbar
            if not(colorbarData is None or oneColorbar is True):
                colorbar = fig.colorbar(colorbarData, ax=axes, **colorbarKwargs)
                if clabel is not None:
                    colorbar.ax.set_title(clabel, rotation=0)

        if oneColorbar is True:
            colorbar = fig.colorbar(colorbarData, ax=ax.ravel().tolist(), **colorbarKwargs)
            if clabel is not None:
                colorbar.ax.set_title(clabel, rotation=0)

        if tight_layout:
            plt.tight_layout()

    else:
        def innerLoop(
                i: int,
                func: Callable,
                rows: int,
                cols: int,
                figSize: tuple
            ):

            tempfig, tempax = plt.subplots(rows, cols, figsize=figSize)
            # Compute
            tempax, colorbarData = func(i,tempax.flatten()[i])

            return [tempfig, tempax, colorbarData]

        generator = (delayed(innerLoop)(i, func, rows, cols, figSize) for i in range(nplots))
        result = Parallel(n_jobs=nWorkers)(generator)

        for i, dummyAxes in enumerate(ax.flatten()):
            # Delete unused axes
            fig.delaxes(dummyAxes)
            if i >= nplots: continue

            tempfig, axes, colorbarData = result[i]
            tempfig.delaxes(axes)   # Unlink and removes axes from temporary figure
            plt.close(tempfig)      # Close temporary figure
            axes.figure = fig       # Link axes to definitive figure
            fig.axes.append(axes)   # Adds axes to definitive figure
            fig.add_axes(axes)      # ??

            if isinstance(xlabel, str):
                if i >= nplots - cols: axes.set_xlabel(xlabel)
                else: axes.set_xlabel('')
            elif hasattr(xlabel, '__iter__'):
                axes.set_xlabel(xlabel[i])

            if isinstance(ylabel, str):
                if i % cols == 0: axes.set_ylabel(ylabel)
                else: axes.set_ylabel('')
            elif hasattr(ylabel, '__iter__'):
                axes.set_ylabel(ylabel[i])

            # Colorbar
            if colorbarData is not None:
                colorbar = fig.colorbar(colorbarData, ax=axes)
                if clabel is not None:
                    colorbar.ax.set_title(clabel, rotation=0)
        plt.tight_layout()

    return fig

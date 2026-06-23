'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
import scipy.stats as st
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from itertools import groupby
from joblib import Parallel, delayed

from validationlib.misc.subsampling import maskApply
from ..plots import probplot, doubleHistogram, coloredBarplot

ALL_DISTS = [
    st.alpha,
    st.anglit,
    st.arcsine,
    st.beta,
    st.betaprime,
    st.bradford,
    st.burr,
    st.cauchy,
    st.chi,
    st.chi2,
    st.cosine,
    st.dgamma,
    st.dweibull,
    st.erlang,
    st.expon,
    st.exponnorm,
    st.exponweib,
    st.exponpow,
    st.f,
    st.fatiguelife,
    st.fisk,
    st.foldcauchy,
    st.foldnorm,
    st.genlogistic,
    st.genpareto,
    st.gennorm,
    st.genexpon,
    st.genextreme,
    st.gamma,
    st.gengamma,
    st.genhalflogistic,
    st.gibrat,
    st.gompertz,
    st.gumbel_r,
    st.gumbel_l,
    st.halfcauchy,
    st.halflogistic,
    st.halfnorm,
    st.halfgennorm,
    st.hypsecant,
    st.invgamma,
    st.invgauss,
    st.invweibull,
    st.johnsonsb,
    st.johnsonsu,
    st.ksone,
    st.kstwobign,
    st.laplace,
    st.levy,
    st.levy_l,
    st.logistic,
    st.loggamma,
    st.loglaplace,
    st.lognorm,
    st.lomax,
    st.maxwell,
    st.mielke,
    st.nakagami,
    st.ncx2,
    st.ncf,
    st.nct,
    st.norm,
    st.pareto,
    st.pearson3,
    st.powerlaw,
    st.powerlognorm,
    st.powernorm,
    st.reciprocal,
    st.rayleigh,
    st.rice,
    st.recipinvgauss,
    st.semicircular,
    st.t,
    st.triang,
    st.truncexpon,
    st.truncnorm,
    st.tukeylambda,
    st.uniform,
    st.vonmises,
    st.vonmises_line,
    st.wald,
    st.weibull_min,
    st.weibull_max,
]
COMMON_DISTS = [st.beta, st.norm, st.johnsonsb, st.johnsonsu, st.logistic, st.cauchy, st.laplace]
PROBLEMATIC_DISTS = [st.gausshyper, st.wrapcauchy, st.rdist, st.levy_stable]
DEFAULT_DISTS = [st.norm, st.laplace, st.cauchy, st.johnsonsu]

def list_parameters(st_dist: Callable) -> List["str"]:
    """
    List parameters for scipy.stats.distribution
    """
    if st_dist.shapes:
        parameters = [name.strip() for name in st_dist.shapes.split(",")]
    else:
        parameters = []
    if st_dist.name in st._discrete_distns._distn_names:
        parameters += ["loc"]
    elif st_dist.name in st._continuous_distns._distn_names:
        parameters += ["loc", "scale"]
    else:
        raise ValueError("Distribution name not found in discrete or continuous lists")
    return parameters

def dist_similarity(*distlist, test="KS", report=False, **kwargs):
    """
        ## **2 sample similarity test**
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

        #### Additional inputs
        ###### kwargs
            Additional arguments for the scipy function
        ###### report: bool

        #### Prints
            If report==True, the output is explained with a short phrase

        #### Outputs
        ###### Statistic, p-value
        <hr style="border:4px solid blue"> </hr> <br />
    """
    if test == "AD":
        test = "Ksample Anderson-Darling"
        n = len(distlist[0])
        AD, _, pvalue = st.anderson_ksamp(distlist, **kwargs)
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
        statistic, pvalue = st.ks_2samp(*distlist, **kwargs)
        test = "2sample Kolmogorov-Smirnov"
    elif test == "WL":
        statistic, pvalue = st.wilcoxon(*distlist, **kwargs)
        test = "2sample Wilcoxon"
    elif test == "MW":
        statistic, pvalue = st.mannwhitneyu(*distlist, **kwargs)
        test = "2sample Mann-Whitney U"
    elif test == "KW":
        statistic, pvalue = st.kruskal(*distlist, **kwargs)
        test = "Ksample Kruskal-Wallis"
    elif test == "chi2":        
        # Based on https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
        assert len(distlist) == 2, "Two-sample chi squared test requires two samples"

        test = "Two-sample chi-squared goodness of fit test"

        dist0 = np.array(distlist[0])
        dist1 = np.array(distlist[1])
        categories_0 = np.unique(dist0)
        categories_1 = np.unique(dist1)
        categories = np.unique(np.concatenate([categories_0, categories_1]))

        r = np.array([np.sum(dist0 == c) for c in categories])
        s = np.array([np.sum(dist1 == c) for c in categories])

        k1 = np.sqrt(s.sum() / r.sum())
        k2 = np.sqrt(r.sum() / s.sum())

        statistic = sum(np.square((k1*r - k2*s)) / (r + s))
        dof = r.shape[0] - 1 if len(distlist[0]) == len(distlist[1]) else r.shape[0]

        pvalue = 1 - st.chi2.cdf(statistic, df=dof)
    else:
        raise ValueError("Error: Choose a valid test")

    if report == True:
        print(str(test) + " test results: statistic=" + str(round(statistic, 3)) + ", pvalue=" + str(round(pvalue, 3)))
    return statistic, pvalue

def dist_similarity_table(
        dist1: pd.DataFrame, 
        dist2: pd.DataFrame, 
        mask: Optional[csr_matrix] = None, 
        tests: List[str] = ["KS"], 
        tests_kwargs: dict[dict] = None,
        title: str = None, 
        alpha: float = 0.05,
        colorPair: list = ['green', 'red'],
    ):
    """
    ## **dist_similarity_table**
    Given two distributions (as dataframes), returns a table with the p-values corresponding to each one of their columns.

    #### Inputs
    ###### dist1
    First distribution
    ###### dist2
    Second distribution
    ###### tests
    List of test to perform. By default it is a Kolmogorov-Smirnov test

    #### Additional inputs
    ###### mask
    Mask to apply to the distributions
    ###### tests_kwargs
    Dictionary with additional arguments for the tests. The keys are the test names
    ###### title
    Title of the table
    ###### alpha
    Minimum admisible p-value. Lower values will be colored as red in the table
    """
    df = pd.DataFrame()

    for i, col in enumerate(dist1.columns):
        for test in tests:
            dist1_masked = maskApply(dist1, mask, i)
            dist2_masked = maskApply(dist2, mask, i)
            if tests_kwargs is not None:
                test_kwargs = tests_kwargs.get(test, {})
            else:
                test_kwargs = {}
            statistic, pvalue = dist_similarity(dist1_masked, dist2_masked, test=test, report=False, **test_kwargs)
            df.loc[col, test] = pvalue

    table = df.style.set_caption(title)
    table.format('{:.5f}')
    if colorPair is not None:
        table.map(lambda x : f'background-color: {colorPair[1]}' if x < alpha else f'background-color: {colorPair[0]}')
    table.map(lambda x: 'color: white')

    return table 

def fit_dist(
    datalist: list,
    scipy_distname=st.johnsonsu,
    labels: list = [""],
    test="KS",
    xlabel="Residue",
    nbins=None,
    report=False,
    visualize=False,
    logscale=True,
):
    """
        ## **Fit sample to distribution** OUTDATED DOCS
        Calculates confidence intervals using bootstrapped method

        #### Inputs
        ###### data : list
        ###### scipy_distname : str
            Specify the name of the scipy distribution to fit the sample to, johnsonsu by default

        #### Additional inputs
        ###### report: bool
        ###### visualize: bool
        ###### statistic_label: str
            Name of the statistic we are calculating the CI of

        #### Prints
            If report==True, prints the result of the AD and KS tests
            If visualize==True, plots two double histograms (with and without logscale) of the result

        #### Outputs
        ###### params
            Output of the fit by scipy
        ###### If visualize==True, outputs fig1, fig2 too
        <hr style="border:4px solid blue"> </hr> <br />
    """
    nlabels = len(datalist)
    fittedlist, pvalue, params = [None] * nlabels, [None] * nlabels, [None] * nlabels

    for i in range(nlabels):
        params[i] = scipy_distname.fit(datalist[i])
        arg, loc, scale = params[i][:-2], params[i][-2], params[i][-1]

        try:
            fittedlist[i] = scipy_distname.rvs(size=len(datalist[i]), loc=loc, scale=scale, *arg)
        except:
            print("rvs error")
        try:
            _, pvalue[i] = dist_similarity(datalist[i], fittedlist[i], test=test, report=report)
        except:
            pvalue[i] = 0

    if visualize == False:
        return params, pvalue, fittedlist
    fig1 = probplot(datalist, labels, scipy_distname=scipy_distname, dist_params=params)
    fig2 = doubleHistogram(
        datalist, fittedlist, labels, xlabel=xlabel, logscale=logscale, dist1_name="True", dist2_name=scipy_distname.name, bins=nbins
    )
    return params, pvalue, fig1, fig2


def cutoff_fit_params(
        dist: pd.DataFrame, 
        yDist: pd.DataFrame,
        mask: csr_matrix = None,
        i: int = None,
        scipy_distname=st.johnsonsu, 
        npoints=100):

    data = maskApply(dist, mask, i)
    ytest = maskApply(yDist, mask, i)
    data_label = dist.columns[i]

    param_names = list_parameters(scipy_distname)
    nparams = len(param_names)
    bounds = [min(ytest), max(ytest)]
    upperbounds = np.linspace(bounds[1], bounds[0], npoints)

    param_values = np.empty((npoints, nparams))

    for i in range(npoints):
        data = data[(ytest < upperbounds[i])]
        ytest = ytest[(ytest < upperbounds[i])]
        if len(data) < 2 or len(ytest) < 2:
            continue

        param_values[i, :] = scipy_distname.fit(data)

    nlabels = nparams
    x = int(np.ceil(nlabels / 3))
    fig, ax = plt.subplots(x, 3, figsize=(18, 4.5 * x))

    for i, axes in enumerate(ax.flatten()):
        if i % 3 == 0:
            plt.setp(axes, ylabel="Parameter value")
        if i >= nlabels - 3:
            plt.setp(axes, xlabel="Filter upper bound")

        if i >= nlabels:
            fig.delaxes(axes)
            continue

        axes.plot(upperbounds, param_values[:, i])
        axes.set_title(data_label + " - " + scipy_distname.name + ' "' + param_names[i] + '"')

def fit_barrage(
        dist: pd.DataFrame, 
        mask: csr_matrix = None,
        i: int = None,
        chosenset="DEFAULT", 
        test = "KS",
        n=5
    ):
    """
        ## **Fit sample to a set of known distributions**
        Calculates confidence intervals using bootstrapped method

        #### Inputs
        ###### data : list
        ###### chosenset : str
            ALL, COMMON or RARE

        #### Additional inputs
        ###### n: int
            n best distributions will be shown

        #### Prints
            Barplot with the result of an AD and a KS test on each distribution's fit. The higher, the better

        #### Outputs
        ###### fig1, fig2
        <hr style="border:4px solid blue"> </hr> <br />
    """
    data = maskApply(dist, mask, i)
    data_label = dist.columns[i]

    if chosenset=='COMMON': DIST = COMMON_DISTS
    elif chosenset=='ALL': DIST = ALL_DISTS
    else: DIST = DEFAULT_DISTS

    results = {}
    for distribution in DIST:
        ## Get parameters of distribution
        params = distribution.fit(data)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        ##
        newdist = distribution.rvs(size=len(data), loc=loc, scale=scale, *arg)
        if test == 'AD': _, pvalue = dist_similarity(data, newdist, test="AD")
        else: _, pvalue = dist_similarity(data, newdist, test="KS")

        results[str(distribution)[32:-34]] = round(pvalue, 4)

    results = sorted(results.items(), key=lambda x: x, reverse=True)
    labels, values = [x[0] for x in results[:n]], [x[1] for x in results[:n]]
    if test == 'AD':
        coloredBarplot(labels, values, ylabel="Anderson-Darling test p-value", pvalueline=0.05)
        plt.title(data_label)
        return
    else: 
        coloredBarplot(labels, values, ylabel="Kolmogorov-Smirnov test p-value", pvalueline=0.05)
        plt.title(data_label)
        return


def waldwolfo_runs(X):
    """
        ## **Performs the Wald-Wolfowitz runs test**
        Statistical test that checks the randomness for a two-valued data sequence

        #### Inputs
        ###### X : list
            Two-valued list to perform the test.

        #### Outputs
        ###### statistic : float
        ###### pvalue : float 
            pvalue calculated following the SAS manual in https://support.sas.com/kb/33/092.html
        <hr style="border:4px solid blue"> </hr> <br />
    """

    if len(set(X)) != 2:
        raise ValueError("Input vector must be a two-valued data sequence.")

    n1 = X.count(set(X).pop())
    n2 = len(X) - n1

    runs_expected = 2 * n1 * n2 / (n1 + n2) + 1
    runs = sum(1 for _ in groupby(X))
    std_dev_runs = np.math.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1)))

    statistic = (runs - runs_expected) / std_dev_runs

    pvalue = 2 * st.norm.sf(np.abs(statistic))

    return statistic, pvalue

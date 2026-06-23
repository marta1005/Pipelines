'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional, Tuple

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from ..misc.subsampling import maskApply
from ..tests import interval as intval
from ..tests.dist import dist_similarity, DEFAULT_DISTS 

threshold = np.nextafter(np.float16(0.), np.float16(1.0))

def filterDiffsTable(
    data: pd.DataFrame,
    mask: csr_matrix
): 
    """
    Filter differences table.
    
    :param data: pd.DataFrame
        Input dataframe.
    :param mask: csr_matrix
        Masking matrix.

    :return: pd.DataFrame
        Filtered table with statistics.
    """
    table = pd.DataFrame({'Count': pd.Series(dtype="str"),
                          'Filtered-Count': pd.Series(dtype="str"),
                          "Min": pd.Series(dtype="float"),
                          "Filtered-Min": pd.Series(dtype="float"),
                          "Max": pd.Series(dtype="float"),
                          "Filtered-Max": pd.Series(dtype="float")})

    for i, output in enumerate(data.columns):
        datatemp = maskApply(data, mask, i)
        table.loc[output,'Count'] = str(len(data[output]))
        table.loc[output,'Filtered-Count'] = str(len(datatemp))
        table.loc[output,'Min'] = np.min(data[output])
        table.loc[output,'Filtered-Min'] = np.min(datatemp)
        table.loc[output,'Max'] = np.max(data[output])
        table.loc[output,'Filtered-Max'] = np.max(datatemp)

    return table.style.set_caption('Final review after masking')

def binsTable(
    df: dict, 
    list_bins: dict,
    k: int, 
    title_bins_count: str,
    title_bins_lims: str,
    min_bins: int,
    colorPair: list = ['green', 'red'],
    colorCmap: Callable = plt.cm.Greens
):
    """
    Generate tables for bins.

    :param df: dict
        Input dataframe.
    :param list_bins: dict
        Dictionary of bin values.
    :param k: int
        Number of bins.
    :param title_bins_count: str
        Title for bins count.
    :param title_bins_lims: str
        Title for bins limits.
    :param min_bins: int
        Minimum bins.
    :param colorPair: list, optional
        Color pair, by default ['green', 'red'].
    :param colorCmap: Callable, optional
        Colormap function, by default plt.cm.Greens.

    :return: Tuple[pd.DataFrame, pd.DataFrame]
        Tables for bins count and limits.
    """
    dict_cont = {}
    for col in df.keys():
        dict_cont[col] = {}
        for i in range (1,k+1):
            dict_cont[col][i]= 0

        for j in range(len(df[col])):
            bin = df[col][j]
            dict_cont[col][int(bin)] +=1

    list_df = pd.DataFrame(dict_cont)
    new_index = []
    for i in range(1, k+1):
        new_index.append('Bin '+str(i))
    list_df.index = new_index
    list_df = list_df.transpose()

    bins= pd.DataFrame()
    for out_var in list_bins.keys(): 
        bins_out_var = list_bins[out_var]
        bins[out_var] = [(round(bins_out_var[i],3), round(bins_out_var[i+1],3)) for i in range(len(bins_out_var)-1)]

    new_index = []
    for i in range(1, k+1):
        new_index.append('Bin '+str(i))
    bins.index = new_index
    bins = bins.transpose()

   
    table_bins_count = list_df.style.set_caption(title_bins_count)  
    table_bins_count.background_gradient(cmap=colorCmap, axis=1)
    table_bins_count.map(lambda x : f'background-color: {colorPair[1]}' if x < min_bins else '')

    table_bins_lims = bins.style.set_caption(title_bins_lims)  
    

    return table_bins_count, table_bins_lims

def binsTableCategory(
    df: dict, 
    list_bins: dict,
    title_bins_count: str,
    min_bins: int,
    colorPair: list = ['green', 'red'],
    colorCmap: Callable = plt.cm.Greens
):
    """
    Generate tables for bins.

    :param df: dict
        Input dataframe.
    :param list_bins: dict
        Dictionary of bin values.
    :param title_bins_count: str
        Title for bins count.
    :param min_bins: int
        Minimum bins.
    :param colorPair: list, optional
        Color pair, by default ['green', 'red'].
    :param colorCmap: Callable, optional
        Colormap function, by default plt.cm.Greens.
    :return: pd.DataFrame
        Tables for bins count
    """
    tables = []
    dict_cont = {}
    for col in df.keys():
        dict_cont = {}
        for i, j in enumerate(list_bins[col]):
            dict_cont[j]=len(df[col][i])
        list_df = pd.DataFrame({col: dict_cont})
    
        list_df = list_df.transpose()
  
        table_bins_count = list_df.style.set_caption(title_bins_count + " for " + col)  
        table_bins_count.background_gradient(cmap=colorCmap, axis=1)
        tables += [table_bins_count.map(lambda x : f'background-color: {colorPair[1]}' if x < min_bins else '')  ]

    return tables

def fit_distribution_similarity(
    residue: pd.DataFrame, 
    mask: Optional[csr_matrix] = None,
    test: str = None,
    method: str = "quick",
    robust_kwargs = {},
    dist: list = DEFAULT_DISTS,
    title: str = None,
    colorPair: list = ['green', 'red'],
    plot_fits: bool = False
):
    """
    Calculate distribution similarity.

    For very large samples, the "quick" method is recommended. For smaller samples, the "robust" method is recommended, as it produces more consistent results (but will be much slower for large samples).

    :param residue: pd.DataFrame
        Residue dataframe.
    :param mask: Optional[csr_matrix], optional
        Masking matrix, by default None.
    :param test: str, optional
        Statistical test, by default None. This is used only for the "quick" method.
    :param method: str, optional
        Method, by default "quick".
        "quick" fits the distribution to the data and performs a goodness-of-fit test.
        "robust" uses goodness_of_fit method from scipy.stats.
    :param robust_kwargs: dict, optional
        Kwargs for scipy.stats.goodness_of_fit, by default {}. This is used only with the "robust" method.
    :param dist: list, optional
        Distribution list, by default DEFAULT_DISTS.
    :param title: str, optional
        Title for the table, by default None.
    :param colorPair: list, optional
        Color pair, by default ['green', 'red'].

    :return: Tuple[pd.DataFrame, pd.DataFrame]
        Table and dataframe.
    """
    
    alpha = 0.05
    df = pd.DataFrame()
    for i, col in enumerate(residue.columns):
        data = maskApply(residue, mask, i)
        for distribution in dist:
            if method == "quick":
                try:
                    # Get parameters of distribution
                    params = distribution.fit(data)
                    arg, loc, scale = params[:-2], params[-2], params[-1]
                    # Random sampling from the fitted distribution
                    fitted = distribution.rvs(size=len(data), loc=loc, scale=scale, *arg)
                    # Similarity test between the original sample and the random sample
                    _, pvalue = dist_similarity(data, fitted, test=test, report=False)
                except:
                    pvalue = np.nan
                    Warning(f"Error fitting {distribution.name} to {col}")

            elif method == "robust":
                res = st.goodness_of_fit(distribution, data, **robust_kwargs)
                pvalue = res.pvalue

            df.loc[col, distribution.name.capitalize()] = pvalue

            if plot_fits and pvalue >= alpha:
                if method == "quick":
                    fig, axs = plt.subplots(figsize=(6, 6))
                    axs.hist(data, bins=50, density=True, color='tab:orange', label="Histogram of Data")
                    axs.plot(np.linspace(data.min(), data.max(), 1000), distribution.pdf(np.linspace(data.min(), data.max(), 1000), loc=loc, scale=scale, *arg), color='tab:blue', linestyle='--', label="Fitted Distribution PDF")
                    axs.set_xlabel(col)
                    axs.set_ylabel("PDF")
                    axs.set_title(f"Fitted {distribution.name} and Histogram")

                    fig.suptitle(f"Fit to {distribution.name} - pvalue: {pvalue}")

                    plt.legend()
                    plt.show()

                elif method == "robust":
                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    res.fit_result.plot(axs[0])
                    axs[0].set_xlabel(col)

                    axs[1].hist(np.log10(res.null_distribution), bins=100)
                    axs[1].set_xlabel(f"log10 of {test} statistic under the null hypothesis")
                    axs[1].set_ylabel("Frequency")
                    axs[1].set_title("Histogram of the Monte Carlo null distribution")

                    fig.suptitle(f"Fit to {distribution.name} - pvalue: {pvalue:.3g}")

                    plt.show()

    table = df.style.set_caption(title)
    table.format('{:.3f}')
    if colorPair is not None:
        table.map(lambda x : f'background-color: {colorPair[1]}' if x < alpha else f'background-color: {colorPair[0]}')
    table.map(lambda x: 'color: white')

    return table, df

statistics = {
    "mean": np.mean,
    "median": [np.percentile, {"q": 50}],
    "std": np.std,
    "IQR": st.iqr,
    "kurtosis": st.kurtosis,
    "skewness": st.skew
}

def prediction_stats(
    error: pd.DataFrame,
    mask: csr_matrix = None,
    statistics: dict = statistics,
    precision: int = 2,
    stdist: Callable = st.johnsonsu,
    method: str = "Bootstrap",
    conf: float = 0.95,
    nsim: int = 100,
    simsize: int = 1000,
    countMinMax: bool = True
) -> pd.DataFrame:
    """
    Calculate prediction statistics.

    :param error: pd.DataFrame
        Error dataframe.
    :param mask: csr_matrix
        Masking matrix.
    :param statistics: dict, optional
        Statistics dictionary, by default statistics.
    :param precision: int, optional
        Precision, by default 2.
    :param stdist: Callable, optional
        Standard distribution, by default st.johnsonsu.
    :param method: str, optional
        Method, by default "Bootstrap".
    :param conf: float, optional
        Confidence level, by default 0.95.
    :param nsim: int, optional
        Number of simulations, by default 100.
    :param simsize: int, optional
        Simulation size, by default 1000.
    :param countMinMax: bool, optional
        Flag for counting min-max, by default True.

    :return: pd.DataFrame
        Prediction statistics table.
    """

    sigFormat = lambda x: np.format_float_positional(x, precision=precision,
                                                     unique=False, fractional=False, trim='k')
    # Create table with the "adequate data types" required after pandas 2.xx
    dtypes_table = {}
    if countMinMax:
        dtypes_table.update({'count': pd.Series(dtype="str"),
                              'min': pd.Series(dtype="str"),
                              'max': pd.Series(dtype="str")})
    if method == "Wilson":
        methodAcronym = "(WS)"
    elif method == "Bootstrap":
        methodAcronym = "(BS)"
    elif method == "Credible Interval":
        methodAcronym = "(Cr)"
    for key in statistics:
        dtypes_table.update({key: pd.Series(dtype="str"),
                             methodAcronym + key: pd.Series(dtype="str")})
    table = pd.DataFrame(dtypes_table)
    for i, name in enumerate(error.columns):
        data = maskApply(error, mask, i)
        if countMinMax: table.loc[name,"count"] = str(len(data))
        
        if countMinMax: table.loc[name,"min"] = f"{sigFormat(np.min(data))}"

        if method == "Wilson": # We also use bootstrap for non-percentile statistics
            model = intval.percentileBootstrap(data, conf=conf, nsim=nsim)
        elif method == "Bootstrap":
            model = intval.percentileBootstrap(data, conf=conf, nsim=nsim)
        elif method == "Credible Interval":
            model = intval.equaltailedCrInterval(data, stdist, conf=conf, nsim=nsim, simsize=simsize)

        for key in statistics:
            if type(statistics[key]) == list:
                func, kwargs = statistics[key]
            else:
                func, kwargs = statistics[key], {}
            value = func(data, **kwargs)

            if method == "Wilson":
                # KDF of the mean, 5%, 90%, 95%, 99% percentiles @95%confidence
                if func == np.percentile:
                    interv = intval.KDF(data, conf=conf, **kwargs)
                else:
                    _, interv = model.compute(func, **kwargs)
            elif method == "Bootstrap":
                # Bootstrap of the mean, 5%, 90%, 95%, 99% percentiles @95%confidence
                _, interv = model.compute(func, **kwargs)
            elif method == "Credible Interval":
                # Credible interval of idem
                _, interv = model.compute(func, **kwargs)

            table.loc[name, key] = f"{sigFormat(value)}"
            table.loc[name, methodAcronym + key] = f"<sup>{sigFormat(interv[1])}</sup><sub>{sigFormat(interv[0])}</sub>"

        if countMinMax: table.loc[name, "max"] = f"{sigFormat(np.max(data))}"

    texty = f"Confidence level at {int(conf * 100)}%."
    if method=='Credible Interval':
        texty += f" Cr=Credible Interval (equal-tailed, {stdist.name})"
    elif method=='Wilson':
        texty += f" WS=CI (wilson-score)"
    else:
        texty += f" BS=Confidence Interval (percentile bootstrap)"
    table = table.style.set_caption(texty
        ).set_table_styles([{'selector': 'th','props': [('font-weight', 'bold')]}]
        ).apply(lambda x: ['font-weight: 100' if v.startswith('<sup>') else '' for v in x]
        ).apply(lambda x: ['font-weight: 100' if v.startswith('(') else '' for v in x]
        ).apply(lambda x: ['font-size: 12px' if v.startswith('<sup>') else '' for v in x]
    )

    return table


def append_errormetrics(ytest: pd.DataFrame, res: pd.DataFrame, outputs_ml: List[str]) -> pd.DataFrame:
    """
    Append error metrics to the dataframe.
    ## **Calculate meaningful metrics on the data**
    MAE, RMSE, critical RF by case and max percent error RF by case

    :param ytest: pd.DataFrame
        True outputs dataframe.
    :param res: pd.DataFrame
        Residue dataframe.
    :param outputs_ml: List[str]
        List of output names.

    :return: pd.DataFrame
        Error metrics appended dataframe.
    """
    errordf = pd.DataFrame(ytest.copy())
    ndata = len(errordf)
    nRF = len(outputs_ml)

    # Using vectorization with numpy or pandas instead of loops we shorten the time required to compute by orders of magnitude
    numpy_RF = ytest.to_numpy()
    numpy_res = res.to_numpy()
    numpy_percentres = np.divide(numpy_res, numpy_RF + threshold) * 100
    numpy_RFnames = np.empty_like(numpy_res, dtype=object)
    for i, RF in enumerate(outputs_ml):
        numpy_RFnames[:, i] = RF

    errordf["MAE"] = np.sum(np.abs(numpy_res), axis=1) / nRF
    #errordf["RMSE"] = np.sqrt(np.sum(np.power(numpy_res, 2), axis=1) / nRF)

    rows, columns = list(range(ndata)), np.argmin(abs(numpy_RF), axis=1)
    errordf["crit_RF"] = numpy_RF[rows, columns]
    errordf["crit_res"] = numpy_res[rows, columns]
    errordf["crit_percent"] = numpy_percentres[rows, columns]
    errordf["crit_RFname"] = numpy_RFnames[rows, columns]

    rows, columns = list(range(ndata)), np.argmax(abs(numpy_percentres), axis=1)
    errordf["maxpercent_RF"] = numpy_RF[rows, columns]
    errordf["maxpercent_res"] = numpy_res[rows, columns]
    errordf["maxpercent_percent"] = numpy_percentres[rows, columns]
    errordf["maxpercent_RFname"] = numpy_RFnames[rows, columns]
    errordf.drop(outputs_ml, axis=1, inplace=True)
    return errordf

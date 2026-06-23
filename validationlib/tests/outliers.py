'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union

import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm

from ..misc.subsampling import maskApply

def outliers_gesd(
        x: np.ndarray, 
        outliers: int, 
        alpha: float = 0.05, 
        report: bool = False,
        displayProgress: bool = True
    ) -> List[float]:
    """
    Detect outliers in a dataset using Generalized ESD (Extreme Studentized Deviate) method.
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

    :param x: Input array or dataset for outlier detection.
    :param outliers: Number of outliers to detect.
    :param alpha: Significance level for hypothesis testing (default: 0.05).
    :param report: If True, provides a summary of the outlier detection process (default: False).
    :param displayProgress: If True, displays progress during outlier detection (default: True).
    :return: List containing information on outlier detection:
        - Index 0: Number of outliers detected.
        - Index 1: Number of the last outlier detected.
        - Index 2: Lower bound of outliers.
        - Index 3: Upper bound of outliers.
    """

    rs, ls = np.zeros(outliers, dtype=float), np.zeros(outliers, dtype=float)
    ms = []
    lenms = 0

    data_proc = np.copy(x)
    argsort_index = np.argsort(data_proc)
    data = data_proc[argsort_index]
    n = data_proc.size
    result, last_bool = [0] * 4, False

    disable = ~displayProgress
    for i in tqdm(range(outliers), disable=disable):
        #print(f"{i} outliers tested\t\t\t", end="\r")
        abs_d = np.abs(data_proc - np.mean(data_proc))

        # R-value calculation
        R = np.max(abs_d) / np.std(data_proc, ddof=1)
        rs[i] = R

        # Masked values
        lms = ms if lenms > 0 else []
        newms = lms + np.where(data == data_proc[np.argmax(abs_d)])[0].tolist()
        lenms += 1
        ms = newms

        # Lambdas calculation
        p = 1 - alpha / (2 * (n - i))
        df = n - i - 2
        t_ppr = st.t.ppf(p, df)
        lambd = ((n - i - 1) * t_ppr) / np.sqrt((n - i - 2 + t_ppr ** 2) * (n - i))
        ls[i] = lambd

        if rs[i] > ls[i]:
            result[1] = i + 1
            last_bool = True
        else:
            if last_bool:
                result[2:3] = min(data_proc), max(data_proc)
                break
            result[0] = i + 2

        if result[1] == 0:
            result[0] = 0

        # Remove the observation that maximizes |xi − xmean|
        data_proc = np.delete(data_proc, np.argmax(abs_d))

    if report:
        print(
            f"Fulfilled hypothesis for {result[0]}-{result[1]} outliers\n Test Statistic: {rs[result[1]-1]:.3f}\n Critical value at \
                {alpha}% significance: {ls[result[1]-1]:.3f}"
        )
        print(f" Outlier lower bounds: [{min(x):.3f},{result[2]:.3f}]")
        print(f" Outlier upper bounds: [{result[3]:.3f},{max(x):.3f}]")
    return result


def outliers_johnsonsu(
    dist: pd.DataFrame, 
    mask: csr_matrix = None,
    stdeviations: float = 3, 
    gesd_alpha: float = 0, 
    maxFraction: float = 0.1, 
    displayProgress: bool = False
) -> List[Union[pd.DataFrame, list]]:
    """
    Detect outliers in a dataset assuming a Johnson SU distribution.

    :param dist: DataFrame containing the dataset.
    :param mask: Sparse matrix mask to apply to the dataset.
    :param stdeviations: Number of standard deviations to consider as an outlier (default: 3).
    :param gesd_alpha: Significance level for GESD outlier detection (default: 0).
    :param maxFraction: Maximum fraction of cases to consider as outliers (default: 0.1).
    :param displayProgress: If True, displays progress during outlier detection (default: False).
    :return: List containing information and results of the outlier detection process:
        - Index 0: Summary table of outlier detection results.
        - Index 1: List of outlier bounds for each label in the dataset.
        - Index 2: DataFrame of transformed data assuming Johnson SU distribution.
        - Index 3: Masked matrix after transformation.
    """

    table = pd.DataFrame(index=[], columns=dist.columns, dtype="object")
    nlabels = len(dist.columns)
    normaldists = [None] * nlabels
    outlier_bounds = [None] * nlabels

    for i in range(nlabels):
        data = maskApply(dist, mask, i)

        gamma, delta, xi, lambd = st.johnsonsu.fit(data)
        normaldists[i] = gamma + delta * np.arcsinh((data - xi) / lambd)
        outlier_bounds[i] = [xi + lambd * np.sinh((-stdeviations - gamma) / delta), xi + lambd * np.sinh((stdeviations - gamma) / delta)]
        
        _, pvalue = st.normaltest(normaldists[i])
        noutliers = sum(abs(j) > stdeviations for j in normaldists[i])
        ncases = len(data)
        
        table.loc["st.normaltest pvalue", dist.columns[i]] = "{:.2f}".format(pvalue)
        table.loc["Total number of cases", dist.columns[i]] = ncases
        table.loc[f"Outliers outside {stdeviations} stds", dist.columns[i]] = f"{noutliers} / {noutliers/ncases:.2%}"
        table.loc[f"Lower bound z={-stdeviations}", dist.columns[i]] = "{:.2f}".format(outlier_bounds[i][0])
        table.loc[f"Upper bound z={stdeviations}", dist.columns[i]] = "{:.2f}".format(outlier_bounds[i][1])

        if gesd_alpha > 0:
            gesd_result = outliers_gesd(normaldists[i], int(maxFraction * ncases), alpha=gesd_alpha, displayProgress=displayProgress)
            outlier_boundsGSD = [
                xi + lambd * np.sinh((gesd_result[2] - gamma) / delta),
                xi + lambd * np.sinh((gesd_result[3] - gamma) / delta),
            ]
            table.loc[f"GESD outliers (a={gesd_alpha})", dist.columns[i]] = f"{gesd_result[0]}-{gesd_result[1]} / {gesd_result[1]/ncases:.2%}"
            table.loc["GESD lower bound", dist.columns[i]] = "{:.2f}".format(outlier_boundsGSD[0])
            table.loc["GESD upper bound", dist.columns[i]] = "{:.2f}".format(outlier_boundsGSD[1])
    
    table = table.style.set_caption("Outliers assuming JohnsonSU distribution")

    df_aux = pd.DataFrame(data=[])
    for i, column in enumerate(dist.columns):
        df_normaldist_indices = maskApply(dist, mask, columnIndex=i, to_numpy=False).index
        df_normaldist = pd.DataFrame(data=normaldists[i], index=df_normaldist_indices, columns=[column])
        df_aux = pd.concat([df_aux, df_normaldist], axis=1)

    normaldists = df_aux
    mask = csr_matrix(normaldists.notna().values)
    
    return table, outlier_bounds, normaldists, mask

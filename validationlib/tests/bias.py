'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Callable, Union

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA

from scipy.sparse import csr_matrix

from matplotlib import colormaps
from matplotlib.colors import Colormap, to_hex

from validationlib.misc.subsampling import mask_condense, maskApply, mask_repeat

def zScoreBias(
        data: pd.DataFrame,
        x: List[str],
        y: str,
        statisticFunc: Callable,
        mask: csr_matrix = None,
        info: bool = True
    ):
    """
    Calculate z-score bias.

    :param data: pd.DataFrame
        Input dataframe.
    :param mask: csr_matrix
        Masking matrix.
    :param x: List[str]
        List of x variables.
    :param y: str
        Y variable.
    :param statisticFunc: Callable
        Statistical function.
    :param info: bool, optional
        Show info, by default True.

    :return: pd.DataFrame
        DataFrame with z-score bias.
    """

    if mask is None:
        masked_data = maskApply(data, mask, columnIndex='all', to_numpy=False, row_wise=True)
    else:
        masked_data = maskApply(data, mask_condense(mask), columnIndex='all', to_numpy=False, row_wise=True)

    groups = list(masked_data.groupby(x, group_keys=False)[y].apply(list).values)
    names = list(masked_data.groupby(x).groups.keys())

    statistic = [statisticFunc(i) for i in groups]
    mean, std = np.mean(statistic), np.std(statistic)

    table = np.zeros((len(groups),3))
    bias = [None]*len(groups)
    for i in range(len(groups)):
        if std > 1e-6: 
            zScore = abs(statistic[i]-mean)/std
            if zScore < 1: bias[i] = '-'
            elif zScore < 3: bias[i] = 'Weak'
            else: bias[i] = 'Strong'
        else:
            print("""Standard deviation close to zero (<1e-6).
                  Z score is just the difference""")
            zScore = abs(statistic[i]-mean)
            bias[i] = 'Strong'

        table[i,0] = len(groups[i])
        table[i,1] = statistic[i]
        table[i,2] = zScore

    columns = ['N',statisticFunc.__name__,'Z score']
    table = pd.DataFrame(table, columns=columns, index=names)
    table['N'] = table['N'].astype(int)
    table['Bias'] = bias

    if info:
        print(f'Chosen statistic: {statisticFunc.__name__}\nGroup mean of {statisticFunc.__name__}: {mean:.4f}; Group std of {statisticFunc.__name__}: {std:.4f}\n')
    pd.set_option('display.max_columns', None)
    table.round(decimals=3)
    return table

def marascuillo(data: pd.DataFrame, x: List[str], y: str, pH0: float=0.05):
    """
    Marascuillo statistical test.

    :param data: pd.DataFrame
        Input dataframe.
    :param x: List[str]
        List of x variables.
    :param y: str
        Y variable.
    :param pH0: float, optional
        Expected proportion, by default 0.05.

    :return: Tuple[pd.DataFrame, float, float]
        DataFrame, chi2 statistic, and p-value.
    """
    pH0 = 0.05  # Expected proportion of defectives
    percentile = np.percentile(abs(data[y]), q=100-pH0*100)

    groups = list(data.groupby(x)[y].apply(list).values)
    names = list(data.groupby(x).groups.keys())
    chi2 = []

    table = np.zeros((len(groups),7))
    for i in range(len(names)):
        n = len(groups[i])
        table[i,0] = n

        pk = len([x for x in groups[i] if abs(x)>=percentile])/n
        table[i,1] = pk*n
        table[i,3] = pk
        chi2Contribution = (pk-pH0)**2/pH0*n
        chi2.append(chi2Contribution)
        table[i,5] = chi2Contribution

        table[i,2] = (1-pk)*n
        table[i,4] = 1-pk
        chi2Contribution = (pH0-pk)**2/(1-pH0)*n
        chi2.append(chi2Contribution)
        table[i,6] = chi2Contribution

    columns = ['N','N DEF','N CON','DEF proportion','CON proportion','DEF chi2 term','CON chi2 term']
    table = pd.DataFrame(table, columns=columns, index=names)
    table['N'] = table['N'].astype(int)
    table['N DEF'] = table['N DEF'].astype(int)
    table['N CON'] = table['N CON'].astype(int)

    doF = (len(groups)-1)+(2-1)
    chi2 = sum(chi2)
    pvalue = 1-st.chi2.cdf(chi2, df=doF)

    print(f'Expected proportion: {pH0}\nchi2 statistic: {round(chi2,4)}; p-value: {pvalue}\nDefective cases=DEF, Conforming cases=CON')
    pd.set_option('display.max_columns', None)
    table.round(decimals=3)
    return table, chi2, pvalue

def anovaTests(data: pd.DataFrame, x: List[str], y: List[str], info: bool=True):
    """
    Perform ANOVA tests.
    If p < 0.05, H0 can be denied with 95% confidence. A fullfilled H0 signifies "All means are equal":
        If p < 0.05, deny H0, deny all means are equal ==> **Error bias if p < 0.05**
        
        If p > 0.05, can't deny H0 ==> Inconclusive

    #### Inputs
    ###### data : pd.DataFrame
        Must contain (at least) the wanted categories
    ###### x: List[str]
        Names of the categorical factors X
    ###### y: List[str]
        Names of the numeric dependent variables Y

    #### Outputs
    ###### test: str
        1-way ANOVA - Compares the groups of variable X (categorical, 1 variable) with Y (numerical, 1 variables)
        2-way ANOVA - Compares the groups of variable X (categorical, 2 variable) with Y (numerical, 2 variables)
        3-way ANOVA - Compares the groups of variable X (categorical, 3 variable) with Y (numerical, 3 variables)
        1-way MANOVA - Compares the groups of variable X (categorical, 1 variable) with Y (numerical, n variables)
        2-way MANOVA - Compares the groups of variable X (categorical, 2 variable) with Y (numerical, n variables)
        3-way MANOVA - Compares the groups of variable X (categorical, 3 variable) with Y (numerical, n variables)
    ###### result: table

    :param data: pd.DataFrame
        Input dataframe.
    :param x: List[str]
        List of x variables.
    :param y: List[str]
        List of y variables.
    :param info: bool, optional
        Show info, by default True.

    :return: pd.DataFrame
        DataFrame with ANOVA results.
    """

    # Format column names according to statmodels requirements
    fmt_cols = {col: col.replace(".", "_").replace(" ", "_") for col in data.columns}
    inv_fmt_cols_mapping = {fmt_col : col for col, fmt_col in fmt_cols.items()}
    data.rename(columns=fmt_cols, inplace=True)

    x_fmt = [fmt_cols[col] for col in x]
    y_fmt = [fmt_cols[col] for col in y]

    functions = {
        'ANOVA': lambda string, data: sm.stats.anova_lm(ols(string, data=data).fit(), typ=2),
        'MANOVA': lambda string, data: MANOVA.from_formula(string, data=data).mv_test()
    }

    # Factors
    nFactors = len(x_fmt)
    if nFactors==1:
        name = '1-way'
        X = x_fmt[0]
    elif nFactors==2:
        name = '2-way'
        X = f'{x_fmt[0]} + {x_fmt[1]} + {x_fmt[0]}:{x_fmt[1]}'
    elif nFactors==3:
        name = '3-way'
        X = f'{x_fmt[0]} + {x_fmt[1]} + {x_fmt[2]} + {x_fmt[0]}:{x_fmt[1]} + {x_fmt[0]}:{x_fmt[2]} + {x_fmt[1]}:{x_fmt[2]} + {x_fmt[0]}:{x_fmt[1]}:{x_fmt[2]}'
    else: raise ValueError('Number of factors different than [1,2,3] not supported')

    # Dependent variables
    nDepend = len(y_fmt)
    Y = " + ".join(str(elem) for elem in y_fmt)
    if nDepend==1:
        name += ' ANOVA'
        func = functions['ANOVA']
    elif nDepend>1:
        name += f' MANOVA ({nDepend} dependent variables)'
        func = functions['MANOVA']
    else: raise ValueError('Number of dependent variables cannot be 0')

    # Covariates (nothing yet until we access or implement ancova/mancova)

    string = f'{Y} ~ {X}'
    if info: print(f'{name}::\t\n{string}')

    # Checks
    for cat in x_fmt + y_fmt:
        if cat not in data.columns: raise ValueError(f'{cat} not present in dataset')
    for cat in x_fmt:
        ngroups = data[cat].nunique()
        if ngroups > 1000: print(f'Warning - {ngroups} groups found in factor "{cat}"')

    result = func(string, data)  
    idx = {x_fmt[i] : x[i] for i in range(len(x))}
    result.rename(index=idx, inplace=True)

    data.rename(columns=inv_fmt_cols_mapping, inplace=True)

    return result


def bias_detection_table(
        data: pd.DataFrame,
        x: List[str],
        y: List[str],
        method:str="anova",
        mask: csr_matrix = None,
        info: bool = True,
        alpha: float = 0.05,
        colorPair: list = ['green', 'red']
    ):
    """
    Generate a table for bias detection.
    The method is either 1-way ANOVA or Kruskal-Wallis H-test.

    :param data: pd.DataFrame
        Input dataframe.
    :param mask: csr_matrix
        Masking matrix.
    :param x: List[str]
        List of x variables.
    :param y: List[str]
        List of y variables.
    :param method: str, optional
        Method, by default "anova". Options are "anova" or "kruskal".
    :param info: bool, optional
        Show info, by default True.
    :param alpha: float, optional
        Alpha value, by default 0.05.
    :param colorPair: list, optional
        Color pair, by default ['green', 'red'].

    :return: pd.DataFrame
        DataFrame with test results.
    """
    assert method in ["anova", "kruskal"], "Method must be 'anova' or 'kruskal'"

    df = pd.DataFrame()

    for j, out_var in enumerate(y):
        if mask is None:
            masked_data = maskApply(data, mask, columnIndex='all', to_numpy=False, row_wise=True)
        else:
            masked_data = maskApply(data, mask[:, j], columnIndex='all', to_numpy=False, row_wise=True)

        for in_var in x:
            if method == "anova":
                table_test = anovaTests(data=masked_data, x=[in_var], y=[out_var], info=info)
                df.loc[in_var, out_var] = table_test.loc[in_var, "PR(>F)"]
            elif method == "kruskal":
                test_results = st.kruskal(*list(masked_data.groupby(in_var)[out_var].apply(list)))
                df.loc[in_var, out_var] = test_results.pvalue

    if method=="anova":
        title = "1-way ANOVA (p-value)" 
    elif method=="kruskal":
        title = "Kruskall-Wallis test (p-value)"

    results_table = df.style.set_caption(title)
    results_table.format('{:.5f}')
    results_table.map(lambda x : f'background-color: {colorPair[0]}' if x >= alpha else f'background-color: {colorPair[1]}')
    results_table.map(lambda x: 'color: white')

    return results_table


def levene_table(
    data: pd.DataFrame,    
    x: List[str],
    y: List[str],
    mask: csr_matrix = None,
    title: str = "Levene test (p-value)",
    alpha: float = 0.05,
    colorPair: list = ['green', 'red']
):
    """
    Generate a table for Levene test.

    :param data: pd.DataFrame
        Input dataframe.
    :param mask: csr_matrix
        Masking matrix.
    :param x: List[str]
        List of x variables.
    :param y: List[str]
        List of y variables.
    :param info: bool, optional
        Show info, by default True.
    :param title: str, optional
        Table title, by default "Levene test (p-value)".
    :param alpha: float, optional
        Alpha value, by default 0.05.
    :param colorPair: list, optional
        Color pair, by default ['green', 'red'].

    :return: pd.DataFrame
        DataFrame with ANOVA results.
    """
    df = pd.DataFrame()

    for i, in_var in enumerate(x):
        for j, out_var in enumerate(y):
            masked_data = list(maskApply(
                data, 
                mask if mask is None else mask[:,j], 
                columnIndex='all', 
                to_numpy=False
            ).groupby(in_var)[out_var].apply(list))

            df.loc[in_var, out_var] = st.levene(*masked_data)[1]

    results_table = df.style.set_caption(title)
    results_table.format('{:.5f}')
    results_table.map(lambda x : f'background-color: {colorPair[0]}' if x >= alpha else f'background-color: {colorPair[1]}')
    results_table.map(lambda x: 'color: white')

    return results_table


def normality_test_table(
    data: pd.DataFrame,
    x: List[str],
    y: List[str],    
    mask: csr_matrix = None,
    title: str = "Normality test (# of bins with p-value over 0.05)",
    alpha: float = 0.05,
    min_bin_size: int = 20,
    cmap: Union[str, Colormap] = "viridis",   
):
    """
    Generate a table for Normality test.

    :param data: pd.DataFrame
        Input dataframe.
    :param mask: csr_matrix
        Masking matrix.
    :param x: List[str]
        List of x variables.
    :param y: List[str]
        List of y variables.
    :param info: bool, optional
        Show info, by default True.
    :param title: str, optional
        Table title, by default "Normality test (# of bins with p-value over 0.05)".
    :param alpha: float, optional
        Alpha value, by default 0.05.
    :param colorPair: list, optional
        Color pair, by default ['green', 'red'].
    :param min_bin_size: int, optional
        Minimum bin size to perform normality test, by default 20.
        If bin has less than min_bin_size, it will be count as non-normal.

    :return: pd.DataFrame
        DataFrame with ANOVA results.
    """
    if isinstance(cmap, str):
        cmap = colormaps.get_cmap(cmap)

    df = pd.DataFrame()

    for i, in_var in enumerate(x):
        for j, out_var in enumerate(y):
            masked_data = list(maskApply(
                data, 
                mask if mask is None else mask[:,j], 
                columnIndex='all', 
                to_numpy=False
            ).groupby(in_var)[out_var].apply(list))

            # For each category, check if the data is normal
            temp = 0
            for k in range(len(masked_data)):
                if len(masked_data[k]) < min_bin_size: continue 
                if st.normaltest(masked_data[k])[1] >= alpha: temp += 1

            df.loc[in_var, out_var] = f'{temp}/{len(masked_data)}' # Proportion of categories with normal data

    results_table = df.style.set_caption(title)
    results_table.map(lambda x : f'background-color: {to_hex(cmap(float(x.split("/")[0])/float(x.split("/")[1])))}')
    results_table.map(lambda x: 'color: white')

    return results_table


def parametric_bias_quantification_pipeline(
    data: pd.DataFrame,
    x: List[str],
    y: List[str],
    mask: csr_matrix = None,
    alpha: float = 0.05,
    min_bin_size: int = 20,
    min_normal_proportion: float = 0.9,
    foolproof: bool = True,
    colorPair: list = ['green', 'red'],
    cmap: Union[str, Colormap] = "viridis"
):
    """
    Bias quantification pipeline. 
    The pipeline consists of:
    1. Levene test
    2. Normality test
    3. ANOVA test
    4. Bias quantification based on mean and variance

    :param data: pd.DataFrame
        Input dataframe.
    :param mask: csr_matrix
        Masking matrix.
    :param x: List[str]
        List of x variables.
    :param y: List[str]
        List of y variables.
    :param alpha: float, optional
        Alpha value, by default 0.05.
    :param min_bin_size: int, optional
        Minimum bin size to perform normality test, by default 20.
        If bin has less than min_bin_size, it will be count as non-normal.
    :param min_normal_proportion: float, optional
        Minimum proportion of bins with normal data to consider the data as normal, by default 0.9.
    :param foolproof: bool, optional
        If True, the ANOVA results will be set to NaN if the data is not homocedastic and normal, by default True.
        If False, the ANOVA results will be kept. These results must be interpreted with caution.
    :param colorPair: list, optional
        Color pair, by default ['green', 'red'].
    :param cmap: Union[str, Colormap], optional
        Colormap, by default "viridis".

    :return: dict
        Dictionary with the following keys
        - levene_results: pd.DataFrame
            DataFrame with Levene results.
        - normality_results: pd.DataFrame
            DataFrame with Normality results.
        - anova_results: pd.DataFrame
            DataFrame with ANOVA results.
        - zscore_tables_mean: List[pd.DataFrame]
            List of DataFrames with z-score bias based on mean.
        - zscore_tables_var: List[pd.DataFrame]
            List of DataFrames with z-score bias based on variance.
    """

    if isinstance(cmap, str):
        cmap = colormaps.get_cmap(cmap)

    # Levene test
    levene_results = levene_table(
        data,
        x,
        y,
        mask,
        alpha=alpha,
        colorPair=colorPair
    )
    homocedastic_inputs = []
    for out_var in y:
        for in_var in x:
            is_homocedastic = levene_results.data.loc[in_var, out_var] >= alpha
            if is_homocedastic: homocedastic_inputs.append((in_var, out_var))

    # Normality test
    normality_results = normality_test_table(
        data,
        x,
        y,
        mask,
        alpha=alpha,
        min_bin_size=min_bin_size,
        cmap=cmap
    )
    normal_inputs = []
    for out_var in y:
        for in_var in x:
            is_normal = int(normality_results.data.loc[in_var, out_var].split("/")[0]) / int(normality_results.data.loc[in_var, out_var].split("/")[1]) >= min_normal_proportion
            if is_normal: normal_inputs.append((in_var, out_var))

    # ANOVA (only when data is "suficiently normal" and homocedastic)
    anova_results = bias_detection_table(
        data,
        x=x,
        y=y,
        method="anova",
        mask=mask,
        info=False
    )

    biased_inputs = []
    for out_var in y:
        for in_var in x:
            anova_assumptions_check = (in_var, out_var) in homocedastic_inputs and (in_var, out_var) in normal_inputs
            if anova_assumptions_check:
                biased = anova_results.data.loc[in_var, out_var] < alpha
                if biased:
                    biased_inputs.append((in_var, out_var))

            else:
                if foolproof:
                    anova_results.data.loc[in_var, out_var] = np.nan

    # Bias quantification based on mean and variance
    zscore_tables_mean = []
    zscore_tables_var = []
    for in_var, out_var in biased_inputs:
        zscore_table_mean = zScoreBias(
            data,
            mask,
            x=in_var,
            y=out_var,
            statisticFunc=np.mean,
            info=False
        )
        zscore_tables_mean.append(zscore_table_mean)

        zscore_table_var = zScoreBias(
            data,
            mask,
            x=in_var,
            y=out_var,
            statisticFunc=np.var,
            info=False
        )
        zscore_tables_var.append(zscore_table_var)

    output_dict = {
        "levene_results": levene_results,
        "normality_results": normality_results,
        "anova_results": anova_results,
        "zscore_tables_mean": zscore_tables_mean,
        "zscore_tables_var": zscore_tables_var
    }

    return output_dict


def trend_table(
        x_data:pd.DataFrame,
        y_data:pd.DataFrame,
        mask:csr_matrix=None,
        normalized_slope:bool=True,
        multiple_tables:bool=True,
        method:str="pearson"
):
    """
    Generate trend tables.

    :param x_data: pd.DataFrame
        X data.
    :param y_data: pd.DataFrame
        Y data.
    :param mask: csr_matrix
        Masking matrix.
    :param normalized_slope: bool, optional
        Use normalized slope, by default True.
    :param multiple_tables: bool, optional
        Generate multiple tables, by default True.

    :return: List[pd.DataFrame]
        List of DataFrames with trend tables.
    """

    assert method in ["pearson", "spearman"], "Method must be 'pearson' or 'spearman'"

    if normalized_slope:
        normalized_str = " (normalized)"
    else:
        normalized_str = ""

    if mask is not None:
        if mask.shape[1] == 1 and y_data.shape[1] > 1:
            mask = mask_repeat(mask, y_data.shape[1])
        elif mask.shape[1] != y_data.shape[1]:
            raise ValueError("The number of columns of the mask must be 1 or the number of columns of the y_data")

    out_tables = []

    tables = pd.DataFrame()
    for j, y_var in enumerate(y_data.columns):
        masked_y = maskApply(y_data, mask, columnIndex=j, to_numpy=False)
        masked_x = maskApply(x_data, mask if mask is None else mask_repeat(mask[:, j], len(x_data.columns)), columnIndex='all', to_numpy=False)

        for x_var in x_data.columns:
            if not multiple_tables and y_var != x_var: continue

            if method == "pearson":
                r, pvalue = st.pearsonr(masked_x[x_var], masked_y)
                m, _ = np.polyfit(masked_x[x_var], masked_y, deg=1)
                if normalized_slope:
                    m = m * (np.max(masked_x[x_var]) - np.min(masked_x[x_var])) / (np.max(masked_y) - np.min(masked_y))

            elif method == "spearman":
                r, pvalue = st.spearmanr(masked_x[x_var], masked_y)

            tables.loc[x_var, method.capitalize() + " coeff."] = r
            tables.loc[x_var, "p-value"] = pvalue

            if method == "pearson":
                tables.loc[x_var, "Slope" + normalized_str] = m

        tables_style = tables.T.copy()
        if multiple_tables:
            out_tables.append(tables_style.style.set_caption(y_var))
        else:
            out_tables = [tables_style.style.set_caption(y_var)]

    return out_tables

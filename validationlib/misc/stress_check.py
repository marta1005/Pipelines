'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
"""
   This module is only for obtaining critical RFs per failure mode and 
   Structural Element. 
   Tested on Aero2Stress project. 
   Example is Stringer & RIB, or Stringer & Frame.
   For other adaptation test it.
"""
import math
from typing import List, Dict, Any, Union
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
def identify_critical_rows(data: pd.DataFrame,
                           id_columns: List[str],
                           target_columns: List[str]) -> Dict[str, List[Any]]:
    """
    Identify critical RFs (minimum) per all id_columns and targets

    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.
    id_columns : List[str]
        DESCRIPTION.
    target_columns : List[str]
        DESCRIPTION.

    Returns
    -------
    Dict[str, List[Any]]
        DESCRIPTION.

    """
    indexes: Dict[str, List[Any]] = {}

    if data.empty or len(id_columns) == 0 or len(target_columns) == 0:
        print("Not enough information. Empty data frame or id_columns or target_columns")
        return indexes

    columns = set(data.columns)
    ids =  set(id_columns)
    if not ids.issubset(columns):
        print("Not enough information. id_columns not in dataframe columns")
        return indexes

    targets =  set(target_columns)
    if not targets.issubset(columns):
        print("Not enough information. targets_columns not in dataframe columns")
        return indexes

    if data[id_columns].empty or data[target_columns].empty:
        print("Not enough information. Empty data frame for id_columns or target_columns")
        return indexes

    for name, group in data.groupby(by=id_columns):
        idx = group[target_columns].idxmin(axis=0)
        indexes.update({name: idx})
    return indexes

def get_critical_rf_check(id_columns: List[str],
                          load_case_columns: Union[str, List[str], int],
                          targets: List[str],
                          test_data: pd.DataFrame,
                          y_hat: pd.DataFrame) -> pd.DataFrame:
    """
    Identify critical RFs (minimum)

    Parameters
    ----------
    id_columns : List[str]
        DESCRIPTION.
    load_case_columns : Union[str, List[str]]
        DESCRIPTION.
    targets : List[str]
        DESCRIPTION.
    test_data : pd.DataFrame
        DESCRIPTION.
    y_hat : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    critical_by_rf : TYPE
        DESCRIPTION.
    summary_report : TYPE
        DESCRIPTION.

    """
    indexes = identify_critical_rows(test_data.reset_index(drop=True),
                                     id_columns, targets)
    dict_df = {k: [] for k in id_columns}
    if load_case_columns is not None:
        if not isinstance(load_case_columns, list):
            load_case_columns = [load_case_columns]
        dict_df.update({i: [] for i in load_case_columns})

    dict_df.update({"Failure Mode": [], "True Value": [], "Predicted Value": [],
                    "Residual": [], "Percentage Error(%)": []})

    for k, v in indexes.items():
        for i, target in enumerate(targets):
            try:
                pos = v[target]
                pred = y_hat[target].iloc[pos]
                true_value = test_data[target].iloc[pos]
                # Postion Columns
                for i, name in enumerate(id_columns):
                    dict_df[name] += [k[i]]
                if load_case_columns is not None:
                    for i, name in enumerate(load_case_columns):
                        dict_df[name] += [test_data[name].iloc[pos]]
                #
                dict_df['Failure Mode'] += [target]
                dict_df['True Value'] += [true_value]
                dict_df['Predicted Value'] += [pred]
                dict_df['Residual'] += [true_value - pred]
                dict_df['Percentage Error(%)'] += [(true_value / pred - 1) * 100]
            except Exception as e:
                print(f"Strural Assembly Position: {k} Min RF Pos:{v}"
                      "\n RF Nº: {i} RF: {target} Data Frame Position: {pos}")
                print(f"Error: {e}")
    critical_by_rf = pd.DataFrame.from_dict(dict_df)
    rows = []
    for name, df in critical_by_rf.groupby(by=id_columns):
        pos = df['True Value'].idxmin()
        rows += [critical_by_rf.iloc[pos:pos+1, :]]
    summary_report = pd.concat(rows, axis=0)
    return critical_by_rf, summary_report

### Display
# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val > 0 else 'black'
    return f'color: {color}'

def highlight_max(s, color: str= 'yellow'):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: ' + color if v else '' for v in is_max]

def highlight_min(s, color: str= 'yellow'):
    '''
    highlight the minimum in a Series blue
    '''
    is_max = s == s.min()
    return ['background-color: ' + color if v else '' for v in is_max]

def dataframe_style(df: pd.DataFrame):
    """
    Predetermined dataframe style.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    style : TYPE
        DESCRIPTION.

    """
    style = df.style.format("{:.3f}",
                            subset=["True Value", "Predicted Value",
                                    "Residual", "Percentage Error(%)"]).\
                            background_gradient(cmap=plt.cm.get_cmap('RdBu'),
                                                #vmin=-10, vmax=10, To add for pandas 1.0
                                                subset=["Residual", "Percentage Error(%)"],
                                                axis=0).\
                           apply(highlight_min, subset=["True Value", "Predicted Value"],
                                 color='yellow')
    return style

###
def report_to_fishtail(df: pd.DataFrame, id_columns: List[str],
                       mapping_column: str,
                       id_1: List[str] = None, id_2:  List[str] = None,
                       show_nan: bool = True) -> pd.DataFrame:
    """
    It create a dataframe where index is id_1 and columns are id_2
    id_columns = [STR, RIB]
    id_1 = [STR1, STR2, ...]
    id_2 = [RIB1, RIB2, ...]

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    id_columns : List[str]
        DESCRIPTION.
    mapping_column : str
        DESCRIPTION.
    id_1 : List[str], optional
        DESCRIPTION. The default is None.
    id_2 : List[str], optional
        DESCRIPTION. The default is None.
    show_nan : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    ft_df : TYPE
        DESCRIPTION.

    """
    if id_1 is None:
        id_1 = (df[id_columns[0]].unique().tolist())
    if id_2 is None:
        id_2 = (df[id_columns[1]].unique().tolist())
    ft_df = pd.DataFrame(index=id_1, columns=id_2).astype(float)
    for _, row in df.iterrows():
        id_x = row[id_columns[0]]
        id_y = row[id_columns[1]]
        if isinstance(row[mapping_column], (float, int)):
            value = row[mapping_column]
        else:
            value = float(row[mapping_column])
        if not math.isnan(value):
            ft_df.loc[id_x, id_y] = value
        else:
            if show_nan:
                print("Not a number in: ", id_x, id_y, "Value: ", row[mapping_column])
    return ft_df

def plot_fishtail(data: pd.DataFrame, title: str, outname: str = None):
    """
    Plot fishtail based on seaborn heatmap
    """
    x_shape, y_shape = data.shape
    if y_shape > x_shape:
        y_shape, x_shape = x_shape, y_shape
        data = data.T
    # Plot axis are different that data frame axis
    y_fig = max(3,  x_shape / 3.)
    x_fig = max(5,  y_shape / 3.)

    _, (ax1, axcb1) = plt.subplots(1, 2,
                                   gridspec_kw={'width_ratios': [1.5, 0.01]},
                                   figsize=(x_fig, y_fig), dpi=150)

    min_value = data.min().min()
    max_value = data.max().max()
    ax_min = max(-10, min_value)
    ax_max = min(10, max_value)

    cmap_error = plt.cm.get_cmap('RdBu')
    if x_fig/y_shape < y_fig/x_shape:
        rotation = 90.
    else:
        rotation = 0.
    # annot=True to get the values on the plot
    ini = sns.heatmap(data, cmap=cmap_error, linewidths=0.1, ax=ax1, cbar=True,
                      cbar_ax=axcb1, center=0.,
                      vmin=ax_min, vmax=ax_max, annot=True, fmt = "2.2f",
                      annot_kws={"size": 8, 'rotation': rotation})
    ini.set_title(title, fontsize='medium')
    ini.tick_params(labelsize=8)
    ini.set_ylabel('', fontsize=8)
    ini.set_xlabel('', fontsize=8)
    plt.tight_layout()

    if outname is not None:
        plt.savefig(outname)
    plt.show();

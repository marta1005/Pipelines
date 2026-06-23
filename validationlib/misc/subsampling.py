'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional, Any

import numbers

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import OrderedDict

threshold = np.nextafter(np.float16(0.), np.float16(1.0))


def bin_data(data, bins : Union[str, int, list] = None, categorical : bool = False, equi_pop: bool = False, include_highest: bool = True):
    """
    Compute bins for a 1D array of data.

    :param data: 1D array-like
        Array of data to bin.
    :param bins: Union[str, int, list], optional
        Specification of the bins. Accepted values are:
        - String: Automatic binning methods accepted by numpy.histogram_bin_edges (see numpy docs).
        - Integer: Number of bins to use.
        - List: List of bin edges.
        - None: In case of categorical data, the unique categories will be used as bins.
    :param categorical: bool, optional
        Whether the data is categorical. If True, the unique categories will be used as bins.
    :param equi_pop: bool, optional
        Whether to use equi-populated bins. If True, the number of bins must be specified.
    :param include_highest: bool, optional
        (For numerical data only) Whether to include the highest value in the last bin. 

    :return: Tuple[np.ndarray, np.ndarray]
        A tuple containing the computed bin boundaries and the indices of the bins.
    """

    data = np.array(data)

    # Checks
    assert data.ndim == 1, "Data must be a 1D array"

    if bins is not None:
        assert np.issubdtype(data.dtype, np.number), "If bins are specified, data must be numeric"

    assert not (bins is None and not categorical), "Either bins or categorical must be specified"

    if equi_pop:
        assert isinstance(bins, int), "Number of bins must be specified for equi-populated bins"

    if categorical:
        categories, bin_indices  = np.unique(data, return_inverse=True)
        return categories, bin_indices
    
    if equi_pop:
        # Compute equi-populated bins
        bin_edges = np.percentile(data, np.linspace(0, 100, bins + 1))
    else:
        # For the rest of the cases, we use numpy.histogram_bin_edges function
        bin_edges = np.histogram_bin_edges(data, bins=bins)

    bin_indices = np.digitize(data, bin_edges)
    if include_highest:
        bin_indices[bin_indices == len(bin_edges)] = len(bin_edges)-1 # Fix last bin
    bin_indices = bin_indices - 1 # Fix zero-based indexing
    return bin_edges, bin_indices



def kPartitions_equi(
    series_bins: list, 
    nbins: int
    ):
    """
    Compute partitions per label for a seri.

    :param series_bins: list
        List of values to partition.
    :param nbins: int
        Number of partitions.

    :return: Tuple[np.ndarray, np.ndarray]
        A tuple containing the computed bin boundaries and the indices of the bins.

    Example:
    >>> bins, binnedIndices = kPartitions_equi([1, 2, 3, 4, 5, 6, 7], 3)
    """
    numeric_values = [value for value in series_bins if not isinstance(value, str)]
    bounds = [min(numeric_values), max(numeric_values)]  
    bins = np.linspace(bounds[0], bounds[1], nbins+1)
    bins[-1] += 0.001
    binnedIndices = np.digitize(series_bins, bins)     
   
    return bins, binnedIndices 

def kPartitions_category(
    series_bins: List[Any]
    ):
    """
    Compute equidistant partitions for a series of values.

    :param series_bins: List[Any]
        List of values to partition.
    :param categories: Any
        Labels to partition.

    :return: Tuple[np.ndarray, np.ndarray]
        A tuple containing the unique categories and the indices of the bins.

    Example:
    >>> bins, binnedIndices = kPartitions_equi([1, 2, 3, 4, 5, 6, 7], 3)
    """
    # Unique values
    categories = list(set(series_bins)) 
    binnedIndices = [np.flatnonzero(series_bins == v) for v in categories]
    
   
    return categories, binnedIndices 

def asign_binnedIndices(
    series: list, 
    binnedIndices: list
    ):  
    """
    Assign values to bins based on computed indices.

    :param series: list
        List of values to assign to bins.
    :param binnedIndices: list
        List of bin indices for the values.

    :return: OrderedDict
        An ordered dictionary where keys are bin indices and values are lists of values in each bin.

    Example:
    >>> series = [1, 2, 3, 4, 5, 6]
    >>> binnedIndices = [1, 2, 1, 3, 3, 2]
    >>> result = asign_binnedIndices(series, binnedIndices)
    """
  
    result = {}
    h = list(set(binnedIndices))
    for bin in h:
        result[bin] = []

    for i in range(len(series)):
        bin = binnedIndices[i]
        result[bin].append(series[i])     
    Result = OrderedDict(sorted(result.items(), key=lambda x: int(x[0])))
        
    return Result


def kPartitions(series, k):
    """
    Compute partitions for a series of values.

    :param series: np.ndarray
        Array of values to partition.
    :param k: int
        Number of partitions.

    :return: np.ndarray
        Array of bin indices for each value.

    Example:
    >>> binnedIndices = kPartitions([1, 2, 3, 4, 5, 6, 7], 3)
    """
    percentiles = np.linspace(0,100,k+1)
    bins = [np.percentile(series, q=j) for j in percentiles]
    bins[-1] += 0.1
    binnedIndices = np.digitize(series, bins)
    
    return binnedIndices


def filterby(comparedSeries: pd.DataFrame, limits: List[float], dataframes: List[pd.DataFrame] = None, return_mask:bool=False) -> List[pd.DataFrame]:
    """
    Filter data based on a comparison between a series of data and a set of limits.

    :param comparedSeries: pd.DataFrame
        Series of data to compare.
    :param limits: List[float]
        [min, max] region of interest.
    :param dataframes: List[pd.DataFrame], optional
        List containing the dataframes to be filtered.
    :param return_mask: bool, optional
        Whether to return the mask instead of the filtered dataframes.

    :return: List[pd.DataFrame]
        List containing the filtered dataframes.

    Example:
    >>> filtered_df1, filtered_df2, filtered_df3 = filterby(series, [0.0, 1.0], [df1, df2, df3])
    >>> mask = filterby(series, [0.0, 1.0], return_mask=True)
    """

    if not return_mask: assert dataframes is not None, "Dataframes must be specified if return_mask is False"

    if return_mask:
        return (comparedSeries > limits[0]) & (comparedSeries < limits[1])

    else:
        output = []
        for x in dataframes:
            output.append(x[(comparedSeries > limits[0]) & (comparedSeries < limits[1])])

        return output

def maskFilter(
        criteriaData: pd.DataFrame, 
        criteria: Union[list, pd.DataFrame], 
        fullperRow: bool=False, 
        criteriaCat: list = None
    ) -> csr_matrix:
    """
    Create a mask for filtering a dataframe based on criteria.

    :param criteriaData: pd.DataFrame
        Dataframe to be masked.
    :param criteria: Union[list, pd.DataFrame]
        List or dataframe with filtering criteria.
    :param fullperRow: bool, optional
        Whether to apply the mask row-wise.
    :param criteriaCat: list, optional
        List of flags indicating categorical variables.

    :return: csr_matrix
        A mask for filtering the dataframe.

    Example:
    >>> mask = maskFilter(df, [0.0, 1.0, 0.5])
    """

    if isinstance(criteria, list):
        bounds = criteria.copy()
        if not isinstance(criteria[0], list):
            bounds = [criteria.copy() for _ in range(len(criteriaData.columns))]
    else:
        bounds = criteria.values.T.tolist()

    mask = criteriaData.copy()

    # Autodetect categorical variables if criteriaCat is not specified. Not recommended when categorical variables are numbers
    if criteriaCat is None:
        criteriaCat = []
        for i in range(len(bounds)):
            if all([isinstance(bounds[i][j], numbers.Number) for j in range(len(bounds[i]))]) and len(bounds[i]) == 2:
                criteriaCat.append(False)
            else:
                criteriaCat.append(True)

    size_criteriaCat = len(criteriaCat)
    for i, column in enumerate(mask):
        if i < size_criteriaCat:
            if criteriaCat[i]:
                mask[column] = mask[column].isin(bounds[i])
            else:            
                mask[column] = (mask[column] >= bounds[i][0]) & (mask[column] <= bounds[i][1])

    if fullperRow:
        mask = np.repeat(mask.all(axis=1).to_numpy().reshape(-1,1),criteriaData.shape[1],axis=1)
    return csr_matrix(mask)

def maskApply(
        data: pd.DataFrame, 
        mask: Optional[csr_matrix] = None,
        columnIndex: int = 0, 
        to_numpy: bool = True,
        np_dtype = None,
        row_wise: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
    """
    Apply a mask to a dataframe and return the filtered data.

    :param data: pd.DataFrame
        Dataframe to be filtered.
    :param mask: Optional[csr_matrix], optional
        A mask for filtering the data.
    :param columnIndex: Union[int, str], optional
        Index or column name to be filtered.
    :param to_numpy: bool, optional
        Whether to return the result as a numpy array.
    :param np_dtype, optional
        The data type to be used when converting to numpy.
    :param row_wise: bool, optional
        Whether to apply the mask row-wise.

    :return: Union[pd.DataFrame, np.ndarray]
        The filtered data based on the mask.

    Example:
    >>> filtered_data = maskApply(df, mask, columnIndex=2, to_numpy=True)
    """

    if mask is None:
        data_ = data.copy()

        if columnIndex != "all":
            data_ = data_.iloc[:,columnIndex]

        if to_numpy: data_ = data_.to_numpy()

        return data_
    
    if row_wise:
        mask = np.repeat(mask.toarray().all(axis=1).reshape(-1,1),data.shape[1],axis=1)

    if columnIndex == "all":
        columnIndex_ = list(np.arange(mask.shape[1]))
        
    else:
        columnIndex_ = columnIndex
    rows, _ = mask[:,columnIndex_].nonzero()
    rows = np.unique(rows)

    if columnIndex == "all" : columnIndex_ = list(np.arange(data.shape[1]))
    if to_numpy: return data.iloc[rows,columnIndex_].to_numpy(dtype=np_dtype)
    
    return data.iloc[rows,columnIndex_]


def mask_and(masks:List[csr_matrix]):
    """
    Compute the element-wise AND of a list of masks.

    :param masks: List[csr_matrix]
        List of masks to be combined using the AND operation.

    :return: csr_matrix
        The resulting mask.

    Example:
    >>> combined_mask = mask_and([mask1, mask2, mask3])
    """
    new_mask = masks[0]
    for i in range(1, len(masks)):
        if masks[i] is not None: new_mask = new_mask.multiply(masks[i])

    return new_mask


def mask_condense(mask:csr_matrix):
    """
    Condense a mask by computing the AND operation row-wise.

    :param mask: csr_matrix
        The mask to be condensed.

    :return: csr_matrix
        The condensed mask.

    Example:
    >>> condensed_mask = mask_condense(mask)
    """
    return csr_matrix(mask.toarray().all(axis=1).reshape(-1, 1))


def mask_repeat(mask:csr_matrix, ncols:int):
    """
    Repeat a mask to match the specified number of columns.

    :param mask: csr_matrix
        The mask to be repeated.
    :param ncols: int
        The number of columns to match.

    :return: csr_matrix
        The repeated mask.

    Example:
    >>> repeated_mask = mask_repeat(mask, 10)
    """
    return csr_matrix(np.repeat(mask.toarray().all(axis=1).reshape(-1,1),ncols,axis=1))

def mask_split(mask:csr_matrix, split_size:float, row_wise=False) -> tuple[csr_matrix]:
    """
    Split a mask into two masks based on the specified split size.

    :param mask: csr_matrix
        The mask to be split.
    :param split_size: float
        The fraction of rows to be included in the first split.
    :param row_wise: bool, optional
        Whether to split row-wise.

    :return: tuple[csr_matrix]
        A tuple containing the two split masks.

    Example:
    >>> split_mask1, split_mask2 = mask_split(mask, 0.7)
    """
    mask1 = np.zeros(mask.shape, dtype=bool)
    mask2 = np.zeros(mask.shape, dtype=bool)

    if row_wise:
        true_idx = np.unique(mask.nonzero()[0])
        indices1 = np.random.choice(true_idx, size=int(split_size*true_idx.shape[0]), replace=False)
        indices2 = true_idx[~np.isin(true_idx, indices1)]

    for i in range(mask.shape[1]):
        if not row_wise:
            true_idx = np.unique(mask[:, i].nonzero()[0])
            indices1 = np.random.choice(true_idx, size=int(split_size*true_idx.shape[0]), replace=False)
            indices2 = true_idx[~np.isin(true_idx, indices1)]
        
        mask1[indices1, i] = True
        mask2[indices2, i] = True

    return csr_matrix(mask1), csr_matrix(mask2)
    

def non_parametric_bootstrap(data: List[float], func: Callable, nsim: int, fraction: float = 1, **kwargs) -> List[float]:
    """
    Perform non-parametric bootstrap resampling.

    :param data: List[float]
        List of data for resampling.
    :param func: Callable
        A function to compute a statistic from the resampled data.
    :param nsim: int
        Number of bootstrap iterations.
    :param fraction: float, optional
        Fraction of the data to use in each iteration.
    :param **kwargs
        Additional arguments for the statistic function.

    :return: List[float]
        A list of computed statistics for each iteration.

    Example:
    >>> resampled_statistics = non_parametric_bootstrap(data, np.mean, 100, fraction=0.8)
    """
    statistic_vector = np.empty(nsim)
    n = len(data)
    simsize = int(n * fraction)
    for i in range(nsim):
        index = np.random.randint(0, n, simsize)
        X = data[index]

        statistic_vector[i] = func(X, **kwargs)  # I compute the statistic and store it

    return statistic_vector


def dfsampling(df: pd.DataFrame, n: Optional[int] = None, fraction: Optional[float] = None) -> pd.DataFrame:
    """
    Perform random sampling without replacement on a dataframe.

    :param df: pd.DataFrame
        Dataframe to be sampled.
    :param n: Optional[int], optional
        Number of samples to take.
    :param fraction: Optional[float], optional
        Fraction of the original sample to use.

    :return: pd.DataFrame
        The sampled dataframe.

    Example:
    >>> sampled_df = dfsampling(df, n=100)
    """

    if fraction == None:
        df = df.sample(n=n, replace=False)
    elif n == None:
        df = df.sample(frac=fraction, replace=False)
    ind = list(range(len(df)))
    df.index = ind

    return df

def outputPartitionsInfo(
    ytest: pd.DataFrame, 
    outputRange: pd.DataFrame, 
    n: int = 5
) -> List[Union[pd.DataFrame, list]]:
    """
    Compute information about output variable partitions.

    :param ytest: pd.DataFrame
        Dataframe containing the output variables.
    :param outputRange: pd.DataFrame
        Dataframe containing the output variable range.
    :param n: int, optional
        Number of partitions.

    :return: List[Union[pd.DataFrame, list]]
        A list containing equidistant partition information, partition boundaries, equipopulated partition information, and partition boundaries.

    Example:
    >>> partition_info = outputPartitionsInfo(ytest, outputRange, n=5)
    """

    equidistTable = pd.DataFrame()
    equipopTable = pd.DataFrame()
    headers = list(ytest.columns.values)

    equidistPartition = np.linspace(min(outputRange.min()), max(outputRange.max()), n)
    equidistPartition = np.round(equidistPartition, 3).tolist()

    # Less than min range
    temp = ytest[ytest < equidistPartition[0]].count()
    for name in headers:
        equidistTable.loc["Smaller than " + str(equidistPartition[0]), name] = temp[
            name
        ]

    for i in range(n - 1):
        interval_name = f"Cases in [{equidistPartition[i]},{equidistPartition[i+1]}]"
        temp = ytest[
            (ytest >= equidistPartition[i]) & (ytest <= equidistPartition[i + 1])
        ].count()

        for name in headers:
            equidistTable.loc[interval_name, name] = temp[name]

    # More than max range
    temp = ytest[ytest > equidistPartition[-1]].count()
    for name in headers:
        equidistTable.loc[f"Larger than {equidistPartition[-1]}", name] = temp[name]

    equidistTable = equidistTable.style.set_caption(
        "Instances organized by evenly spaced intervals"
    )

    equipopPartition = [[0] * n for _ in headers]
    percentiles = np.linspace(0, 100, n)

    for j, name in enumerate(headers):
        temp = ytest[name].copy()
        temp = temp[
            (temp >= outputRange[name]["min"]) & (temp <= outputRange[name]["max"])
        ]
        for i in range(n):
            equipopPartition[j][i] = temp.quantile(percentiles[i] / 100)
            equipopTable.loc[f"Percentile {percentiles[i]}", name] = equipopPartition[
                j
            ][i]
    equipopTable = equipopTable.style.set_caption(
        "Instances inside applicability organized by evenly populated intervals"
    )

    return equidistTable, equidistPartition, equipopTable, equipopPartition
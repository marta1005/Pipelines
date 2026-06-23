'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable

import pandas as pd
import numpy as np


def permutation_test(
    A: Union[pd.DataFrame, np.ndarray],
    B: Union[pd.DataFrame, np.ndarray],
    function: Callable,
    n_permutations: int = 100,
    n_subsample: int = 1000,
    paired: bool = False,
) -> List[Union[float, list]]:
    """
    Perform a permutation test to assess the statistical significance of the difference 
    between two datasets.

    :param A: First dataset (DataFrame or ndarray).
    :param B: Second dataset (DataFrame or ndarray).
    :param function: Function to compute the test statistic between the datasets.
    :param n_permutations: Number of permutations to perform (default: 100).
    :param n_subsample: Number of subsamples to consider in each permutation (default: 1000).
    :param paired: If True, perform a paired permutation test (default: False).
    :return: List containing:
        - The true test statistic between the original datasets.
        - List of test statistics obtained from permutations.
        - The calculated p-value.
    """
    X, Y = A.copy(), B.copy()
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    n_subsample = min(n_subsample, len(X))
    rows = np.random.randint(0, len(X), size=n_subsample)
    X, Y = X[rows, :], Y[rows, :]

    true_score = function(X, Y)

    perm_scores = [None] * n_permutations
    for i in range(n_permutations):
        if paired:
            X_perm, Y_perm = permute_pairs(X, Y, n_subsample)
            perm_scores[i] = function(X_perm, Y_perm)
        else:
            perm_scores[i] = function(permute(X), permute(Y))

    # The pvalue is calculated as: number of permutations with score >= true score / number of permutations
    C = sum(i > true_score for i in perm_scores)
    pvalue = 1 - C / n_permutations

    return true_score, perm_scores, pvalue


def permute(X: np.ndarray) -> np.ndarray:
    """
    Permute the elements of a NumPy array.

    :param X: Input array to permute.
    :return: Permuted array.
    """
    Z = X.copy()
    if X.ndim > 1:
        shap = Z.shape
        Z = Z.flatten()
    Z = np.random.permutation(Z)
    if X.ndim > 1:
        Z = Z.reshape(shap)
    return Z


def permute_pairs(X: np.ndarray, Y: np.ndarray, n_subsample: int) -> List[np.ndarray]:
    """
    Permute pairs of elements in two NumPy arrays.

    :param X: First array to permute.
    :param Y: Second array to permute.
    :param n_subsample: Number of subsamples for permutation.
    :return: List containing permuted versions of input arrays X and Y.
    """
    k = np.random.randint(0, n_subsample)
    swap_rows = np.random.randint(0, n_subsample, size=k)
    X_perm, Y_perm = X.copy(), Y.copy()

    temp = X_perm[swap_rows, :]
    X_perm[swap_rows, :] = Y_perm[swap_rows, :]
    Y_perm[swap_rows, :] = temp

    return X_perm, Y_perm

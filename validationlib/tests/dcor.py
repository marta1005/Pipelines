'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform


def dcor(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the distance correlation between two matrices X and Y.

    :param X: Input matrix X of shape (n_samples, n_features).
    :param Y: Input matrix Y of shape (n_samples, n_features).
    :return: Distance correlation between matrices X and Y.
    """
    assert X.shape[0] == Y.shape[0]

    A = cent_dist(X)
    B = cent_dist(Y)

    dcov_AB = dcov(A, B)
    dvar_A = dvar(A)
    dvar_B = dvar(B)

    dcor = 0.0
    if dvar_A > 0.0 and dvar_B > 0.0:
        dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)

    return dcor


def dcov(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the distance covariance between matrices X and Y.

    :param X: Input matrix X of shape (n_samples, n_features).
    :param Y: Input matrix Y of shape (n_samples, n_features).
    :return: Distance covariance between matrices X and Y.
    """
    n = X.shape[0]
    XY = np.multiply(X, Y)
    cov = np.sqrt(XY.sum()) / n
    return cov


def dvar(X) -> float:
    """
    Computes the distance variance of a matrix X.

    :param X: Input matrix X of shape (n_samples, n_features).
    :return: Distance variance of matrix X.
    """
    return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))


def cent_dist(X: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Euclidean distance between rows of X and centers
    each cell of the distance matrix with row mean, column mean, and grand mean.

    :param X: Input matrix X of shape (n_samples, n_features).
    :return: Centered distance matrix of shape (n_samples, n_samples).
    """
    M = squareform(pdist(X))  # distance matrix
    rmean = M.mean(axis=1)
    cmean = M.mean(axis=0)
    gmean = rmean.mean()
    R = np.tile(rmean, (M.shape[0], 1)).transpose()
    C = np.tile(cmean, (M.shape[1], 1))
    G = np.tile(gmean, M.shape)
    CM = M - R - C + G
    return CM

import os
from typing import List
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from .convexhull import chmp_lp
from ..tests.dist import dist_similarity
from ..tests.interval import percentileBootstrap

@dataclass
class SplitValidationResult: 
    residual_voxel_proportion: float = None
    valid_test_proportion: float = None
    phacking_test_proportion: float = None
    isolated_test_proportion: float = None
    test_outside_ch_proportion: float = None
    isolated_train_proportion: float = None
    chi_squared_pvalue: float = None
    valid_test_point: np.ndarray = None
    phacking_test: np.ndarray = None
    isolated_test: np.ndarray = None
    test_outside_ch: np.ndarray = None
    isolated_train_point: np.ndarray = None

    def __repr__(self):
        return f"SplitValidationResult(residual_voxel_proportion={self.residual_voxel_proportion}, valid_test_proportion={self.valid_test_proportion}, phacking_test_proportion={self.phacking_test_proportion}, isolated_test_proportion={self.isolated_test_proportion}, test_outside_ch_proportion={self.test_outside_ch_proportion}, isolated_train_proportion={self.isolated_train_proportion}, chi_squared_pvalue={self.chi_squared_pvalue})"

    def _repr_pretty_(self, pp, cycle):
        from IPython.display import display
        table = pd.DataFrame(columns=["Results"])
        table.loc["Residual Voxel Proportion", "Results"] = self.residual_voxel_proportion
        table.loc["Valid Test Proportion", "Results"] = self.valid_test_proportion
        table.loc["Phacking Test Proportion", "Results"] = self.phacking_test_proportion
        table.loc["Isolated Test Proportion", "Results"] = self.isolated_test_proportion
        table.loc["Test Outside CH Proportion", "Results"] = self.test_outside_ch_proportion
        table.loc["Isolated Train Proportion", "Results"] = self.isolated_train_proportion
        table.loc["Chi-Squared p-Value", "Results"] = self.chi_squared_pvalue
        display(table)

def voxel_tesselation_proximity_method(
    train_data,
    test_data,
    categorical_variables: List[int] = [],
    k_nearest:int=None,
    compute_chmp:bool=False,
    verbose:bool=False,
    strict_mode:bool=False,
    bootstrap:bool=False,
    confidence:float=0.95,
    nsim_bootstrapping:int=1000
):
    """
    Method to validate a split of data into training and test sets using the Voxel Tesselation Proximity method.

    The method is based on the Voxel Tesselation Proximity method, which divides the data space into voxels according to the categories of the training data. The method then checks the proximity of the test points to the training points within the same voxel. The method also checks for isolated training points.

    :param train_data: The training data as an array-like object.
    :param test_data: The test data as an array-like object.
    :param categorical_variables: A list of column indices of the categorical variables in the data.
    :param k_nearest: The number of nearest neighbors to consider when checking for phacking and isolated points.
    :param compute_chmp: Whether to compute the Convex Hull Membership Probelm (CHMP) on the test points or not.
    :param verbose: Whether to print additional information about the validation process.
    :param strict_mode: Whether to return the validation result as soon as a criterion is not met.
    :param bootstrap: Whether to use bootstrapping to compute the thresholds for phacking, isolated test and isolated training points.
    :param confidence: When bootstrap=True, the confidence level to use when computing the thresholds for phacking, isolated test and isolated training points.
    :param nsim_bootstrapping: When bootstrap=True, the number of simulations to use when computing the thresholds for phacking, isolated test and isolated training points.
    """

    val_result = SplitValidationResult()

    # Cast data to numpy arrays
    train_data = np.array(train_data).copy()
    test_data = np.array(test_data).copy()

    # Parameter checks
    if k_nearest is None:
        k_nearest = train_data.shape[0]

    # Scale the numerical variables
    scaler = MinMaxScaler()
    for i in range(train_data.shape[1]):
        if i not in categorical_variables:
            scaler.fit(np.vstack([train_data[:, i].reshape(-1,1), test_data[:, i].reshape(-1,1)]))
            train_data[:, i] = scaler.transform(train_data[:, i].reshape(-1, 1)).squeeze()
            test_data[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).squeeze()

    # If no categorical variables are provided, assume all samples belong to the same voxel by creating a dummy categorical variable
    if len(categorical_variables) == 0:
        # One voxel to rule them all
        if verbose: print("Dummy categorical created")
        train_data = np.hstack([train_data, np.zeros(train_data.shape[0], dtype=bool)[:, None]])
        test_data = np.hstack([test_data, np.zeros(test_data.shape[0], dtype=bool)[:, None]])
        categorical_variables = [train_data.shape[1]-1]

    categorical_variables = np.array(categorical_variables, dtype=int) 
    numerical_variables = np.array([i for i in range(train_data.shape[1]) if i not in categorical_variables], dtype=int)

    # Create voxels according to train categories
    cats_per_var = [np.unique(train_data[:, var]) for var in categorical_variables] # Categories per variable
    voxels = np.array(list(product(*cats_per_var))).astype(str) # All possible combinations of categories

    # Assign each sample of training and test to a voxel. If some combination of categories is not present in the computed voxels, assign it to the residual voxel.
    train_voxels = np.ones(train_data.shape[0], dtype=int)*-1 # -1 means residual voxel
    test_voxels = np.ones(test_data.shape[0], dtype=int)*-1 # -1 means residual voxel

    for i, voxel in enumerate(voxels):
        train_mask = np.all(train_data[:, categorical_variables].astype(str) == voxel, axis=1)
        test_mask = np.all(test_data[:, categorical_variables].astype(str) == voxel, axis=1)
        train_voxels[train_mask] = i
        if train_mask.any():
            test_voxels[test_mask] = i
        else:
            test_voxels[test_mask] = -1

    # If there are more than 5%% of test points in the residual voxel, flag the split as not valid
    residual_proportion = np.sum(test_voxels == -1)/test_data.shape[0]
    val_result.residual_voxel_proportion = residual_proportion
    if residual_proportion > 0.05:
        if strict_mode:
            return val_result

    if len(voxels) > 1:
        # Compute a chi-squared test comparing the distribution of training and test samples per voxel. If p-value < 0.05, flag split as not valid
        _, pvalue = dist_similarity(train_voxels, test_voxels, test="chi2")
        val_result.chi_squared_pvalue = pvalue
        if pvalue < 0.05:
            if strict_mode:
                return val_result

    phacking_mask = np.zeros(test_data.shape[0], dtype=bool) # True if test point is too close to training set
    isolated_mask = np.zeros(test_data.shape[0], dtype=bool) # True if test point is too far from training set
    outside_ch_mask = np.zeros(test_data.shape[0], dtype=bool) # True if test point is outside the convex hull of the training set
    valid_test_mask = np.zeros(test_data.shape[0], dtype=bool) # True if test point is valid (not phacking, not isolated, not outside the convex hull)
    notest_mask = np.zeros(train_data.shape[0], dtype=bool) # True if training point is too far from any test point

    for i, voxel in enumerate(voxels):
        # Get the indices of the samples in the training and test set that belong to the current voxel
        voxel_train_indices = np.where(train_voxels == i)[0]
        voxel_test_indices = np.where(test_voxels == i)[0]

        # Get the data of the samples in the training and test set that belong to the current voxel
        voxel_train_data = train_data[voxel_train_indices][:, numerical_variables].astype(np.float64)
        voxel_test_data = test_data[voxel_test_indices][:, numerical_variables].astype(np.float64)

        # Skip to the next voxel if there are no samples in the test set that belong to the current voxel
        if len(voxel_test_indices) == 0:
            continue

        # Compute train-test distance matrix (inside a voxel)
        train_test_d = np.linalg.norm(voxel_train_data[:, None] - voxel_test_data[None], axis=-1)
        train_train_d = np.linalg.norm(voxel_train_data[:, None] - voxel_train_data[None], axis=-1)

        # Checks for every test point of the voxel
        for j, test_idx in enumerate(voxel_test_indices):
            # Get the k nearest training points to the current test point
            nearest_train_idx = np.argsort(train_test_d[:, j])[:k_nearest]

            # Compute pairwise distances between the k_nearest training points (of the same voxel)
            kn_tr_tr_d = np.linalg.norm(voxel_train_data[nearest_train_idx, None] - voxel_train_data[None, nearest_train_idx, :], axis=-1)

            # Compute distances between the current test point and the k_nearest training points (of the same voxel)
            kn_tr_te_d = np.linalg.norm(voxel_train_data[nearest_train_idx, None] - voxel_test_data[None, j], axis=-1)

            np.fill_diagonal(kn_tr_tr_d, np.inf) # Remove self-distances
            kn_tr_tr_d_min = np.min(kn_tr_tr_d, axis=1) # Minimum distance between each training point and its k_nearest neighbors

            # Compute the threshold for phacking and isolated test points
            # These thresholds are based on percentiles of the distribution of minimum pairwise distances between training points
            if bootstrap:
                bootstrapper = percentileBootstrap(kn_tr_tr_d_min, conf=confidence, nsim=nsim_bootstrapping)

                thres_d_phacking, _ = bootstrapper.compute(np.quantile, q=0.025)
                thres_d_phacking = 0.5*thres_d_phacking

                thres_d_isolated, _ = bootstrapper.compute(np.quantile, q=0.975)
            else:
                thres_d_phacking = 0.5*np.quantile(kn_tr_tr_d_min, 0.025)
                thres_d_isolated = np.quantile(kn_tr_tr_d_min, 0.975)

            if np.any(kn_tr_te_d < thres_d_phacking):
                phacking_mask[test_idx] = True

            if np.all(kn_tr_te_d > thres_d_isolated):
                isolated_mask[test_idx] = True

            if compute_chmp:
                if not chmp_lp(voxel_train_data, voxel_test_data[j]):
                    outside_ch_mask[test_idx] = True

            # Validity criteria for the current test point depends on whether we are computing the CHMP or not
            if compute_chmp:
                valid_test_mask[test_idx] = not phacking_mask[test_idx] and not isolated_mask[test_idx] and not outside_ch_mask[test_idx]
            else:
                valid_test_mask[test_idx] = not phacking_mask[test_idx] and not isolated_mask[test_idx]

        # Checks for every training point of the voxel
        for j, train_idx in enumerate(voxel_train_indices):
            # Get the k nearest train points to the current training point
            nearest_train_idx = np.argsort(train_train_d[:, j])[:k_nearest]

            kn_tr_tr_d = np.linalg.norm(voxel_train_data[nearest_train_idx, None] - voxel_train_data[None, nearest_train_idx, :], axis=-1)
            kn_tr_te_d_min = np.min(np.linalg.norm(voxel_train_data[nearest_train_idx, None] - voxel_test_data[None, :], axis=-1))

            np.fill_diagonal(kn_tr_tr_d, np.inf) # Remove self-distances
            kn_tr_tr_d_min = np.min(kn_tr_tr_d, axis=1)

            # Compute the threshold for isolated training points
            # This threshold is based on percentiles of the distribution of minimum pairwise distances between training points
            if bootstrap:
                bootstrapper = percentileBootstrap(kn_tr_tr_d_min, conf=confidence, nsim=nsim_bootstrapping)
                thres_d_notest, _ = bootstrapper.compute(np.quantile, q=0.975)
            else:
                thres_d_notest = np.quantile(kn_tr_tr_d_min, 0.975)

            # If closest test point to the current training point is too far, flag the training point as isolated
            if kn_tr_te_d_min > thres_d_notest:
                notest_mask[train_idx] = True

    val_result.phacking_test = phacking_mask
    val_result.isolated_test = isolated_mask
    val_result.test_outside_ch = outside_ch_mask
    val_result.valid_test_point = valid_test_mask
    val_result.isolated_train_point = notest_mask
    val_result.valid_test_proportion = np.sum(valid_test_mask)/test_data.shape[0]
    val_result.phacking_test_proportion = np.sum(phacking_mask)/test_data.shape[0]
    val_result.isolated_test_proportion = np.sum(isolated_mask)/test_data.shape[0]
    val_result.test_outside_ch_proportion = np.sum(outside_ch_mask)/test_data.shape[0]
    val_result.isolated_train_proportion = np.sum(notest_mask)/train_data.shape[0]

    return val_result
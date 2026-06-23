'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
import copy
import itertools
from itertools import cycle
import os.path
import json
from typing import List, Dict, Union, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer

import scipy.stats as st

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors
import seaborn as sns

from joblib import Parallel, delayed

from ..tests import interval
from ..misc.utils import HiddenPrints
from ..plots.common import multiplePlots
import validationlib.models as models

import shap


@dataclass
class PermutationImportanceResults:
    """
    Class to store the results of a permutation importance study.

    :param baseline_score: Score of the model without performing any permutation.
    :param importances_mean: Mean importance of each variable.
    :param importances_std: Standard deviation of the importance of each variable.
    :param importances: The importances for each permutation and variable.
    :param bootstrapped: If the importances are bootstrapped.
    :param importances_bootstrapped: Mean importance of each variable obtained through bootstrapping.
    :param importances_ci: The confidence interval of the importance for each variable.
    :param metric_name: Name of the metric used to compute the importances.
    """
    baseline_score: float
    importances_mean: np.ndarray
    importances_std: np.ndarray
    importances: np.ndarray
    bootstrapped: bool = False
    importances_bootstrapped: np.ndarray = None
    importances_ci: List[np.ndarray] = None
    metric_name: str = "score"


class ImportanceScorer:
    """
    Class to wrap a scoring function for importance studies.

    :param scorer: Scoring function to be used. Must have the same signature as the scoring functions in scikit-learn: (estimator, X, y).
    :param greater_is_better: Whether a higher score is better or not.
    """
    def __init__(self, scorer: Callable, greater_is_better: bool = False):
        self.scorer = scorer
        self.greater_is_better = greater_is_better
    
    def __call__(self, estimator, X, y):
        return self.scorer(estimator, X, y)

    @classmethod
    def make_scorer(cls, metric: Callable, greater_is_better: bool = False, **kwargs):
        """
        Make a scorer from a metric function. Wrapper around scikit-learn's make_scorer function.

        :param metric: Metric function to be used.
        :param greater_is_better: Whether a higher score is better or not.
        :param kwargs: Additional keyword arguments to be passed to the metric function.
        :return: ImportanceScorer instance.
        """
        scorer = make_scorer(metric, greater_is_better=greater_is_better, **kwargs)

        return cls(scorer, greater_is_better)


def __calculate_permutation_scores(
    estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, col_idx: int, scoring: "ImportanceScorer", n_repeats: int, random_state: int
):
    """
    Calculate permutation scores for a variable.

    This function computes permutation scores for a specific variable using a scikit-learn estimator.
    This function is only for the internal use of the library.
    If you want to compute the permutation importances, please use 'permutation_importance' or 'bootstrapped_permutation_importance' instead.

    :param estimator: A scikit-learn trained BaseEstimator to compute scores from.
    :param X: Input data.
    :param y: Output ground truth.
    :param col_idx: Index of the variable that is going to be permuted.
    :param scoring: Function to obtain the scores in each permutation. Must be obtained through scikit-learn's `make_scorer` function.
    :param n_repeats: Number of permutations.
    :param random_state: Seed for the random permutations.
    :return: A list of scores for each permutation and sample.
    """
    np.random.seed(random_state)

    scores = []
    X_permuted = copy.deepcopy(X)

    for _ in range(n_repeats):
        np.random.shuffle(X_permuted[:, col_idx])
        scores.append(scoring(estimator, X_permuted, y))

    return scores


def permutation_importance(
    estimator: BaseEstimator, X, y, scoring: "ImportanceScorer", n_repeats: int = 5, n_jobs: int = None, random_state: int = None, metric_name: str = None
):
    """
    Calculate permutation importances of input variables in a model.

    This function computes the permutation importance of each input variable in a model using a scikit-learn estimator.

    :param estimator: A scikit-learn trained BaseEstimator to compute importances from.
    :param X: Input data.
    :param y: Output ground truth.
    :param scoring: Function to obtain the scores in each permutation. Must be obtained through scikit-learn's `make_scorer` function.
    :param n_repeats: Number of permutations (default is set to 5).
    :param n_jobs: Number of processes for computing the importances. The 'estimator' must be able to run in parallel with other instances of itself.
    :param random_state: Seed for the random permutations.
    :param metric_name: Name of the metric used to compute the importances.

    :return: A PermutationImportanceResults instance containing the importances.
    """
    X = np.array(X)
    y = np.array(y)
    if n_jobs is None:
        n_jobs = 1

    baseline_score = scoring(estimator, X, y)

    if n_jobs > 1:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(__calculate_permutation_scores)(estimator, X, y, col_idx, scoring, n_repeats, random_state)
            for col_idx in range(X.shape[1])
        )
    else:
        scores = np.array([
            __calculate_permutation_scores(estimator, X, y, col_idx, scoring, n_repeats, random_state) for col_idx in range(X.shape[1])
        ])

    importances = baseline_score - scores
    importances_mean = np.mean(importances, axis=1)
    importances_std = np.std(importances, axis=1)

    pi_results = PermutationImportanceResults(
        baseline_score=baseline_score if scoring.greater_is_better else -baseline_score,
        importances_mean=importances_mean,
        importances_std=importances_std,
        importances=importances,
    )

    if metric_name is not None:
        pi_results.metric_name = metric_name

    return pi_results


def bootstrapped_permutation_importance(
    estimator: BaseEstimator,
    X,
    y,
    scoring: "ImportanceScorer",
    n_repeats: int = 5,
    n_sim: int = 100,
    conf: float = 0.95,
    n_jobs: int = None,
    random_state: int = None,
    metric_name: str = None,
):
    """
    Calculate bootstrapped permutation importances of input variables in a model.

    This function computes the permutation importance of each input variable in a model and returns the mean importances and their confidence interval obtained through bootstrapping.

    :param estimator: A scikit-learn trained BaseEstimator to compute importances from.
    :param X: Input data.
    :param y: Output ground truth.
    :param scoring: Function to obtain the scores in each permutation. Must be obtained through scikit-learn's `make_scorer` function.
    :param n_repeats: Number of permutations (default is set to 5).
    :param n_sim: Number of bootstrapping simulations (default is set to 100).
    :param conf: Confidence for the confidence interval (must be a number between 0 and 1).
    :param n_jobs: Number of processes for computing the importances. The 'estimator' must be able to run in parallel with other instances of itself.
    :param random_state: Seed for the random permutations.
    :param metric_name: Name of the metric used to compute the importances.
    :return: A PermutationImportanceResults instance containing the importances.
    """
    pi_results = permutation_importance(estimator, X, y, scoring=scoring, n_repeats=n_repeats, n_jobs=n_jobs, random_state=random_state, metric_name=metric_name)
    pi_results.bootstrapped = True
    pi_results.importances_bootstrapped = np.zeros_like(pi_results.importances_mean)
    pi_results.importances_ci = []


    for i in range(len(pi_results.importances)):
        bootstrapper = interval.percentileBootstrap(
            pi_results.importances[i], conf=conf, nsim=n_sim, fraction=1, random_state=random_state
        )
        mean, ci_mean = bootstrapper.compute(np.mean)
        pi_results.importances_bootstrapped[i] = mean
        pi_results.importances_ci.append(ci_mean)


    return pi_results


def get_importance_table(importance_results: "PermutationImportanceResults", labels: List[str], title: str = None, bootstrapped: bool = True):
    """
    Generate an importance table for visualization.

    Given a dictionary containing the permutation importance of an estimator, this function returns a table containing the ordered importances.

    :param importances_dict: Dictionary containing the importances, produced by the functions 'permutation_importance' or 'bootstrapped_permutation_importance'.
    :param labels: List of variable names.
    :param title: Title of the table.
    :param bootstrapped: If importance results come from a bootstrapped study, whether or not to use the importance bootstrapped quantities.
    :return: A Pandas DataFrame containing the ordered mean and deviation (or bootstrapped mean and confidence interval) of each variable.
    """
    df = pd.DataFrame()

    if importance_results.bootstrapped and bootstrapped:
        importance_mean = importance_results.importances_bootstrapped
        importance_dev = importance_results.importances_ci

        error_column = "CI"
    else:
        importance_mean = importance_results.importances_mean
        importance_dev = importance_results.importances_std

        error_column = "Importance std"

    importances_order = np.flip(np.argsort(importance_mean))

    for i in range(len(labels)):
        col = labels[importances_order[i]]
        df.loc[col, "Mean importance"] = "%.4E" % importance_mean[importances_order[i]]

        if type(importance_dev[i]) is list:
            error_text = "[" + ", ".join(["%.4E" % num for num in importance_dev[importances_order[i]]]) + "]"
        else:
            error_text = "%.4E" % (importance_dev[importances_order[i]])
        df.loc[col, error_column] = error_text

    title_prefix = "" if title is None else title + " - "
    table = df.style.set_caption(title_prefix + "Baseline %s=%.4E" % (importance_results.metric_name, importance_results.baseline_score))
    return table


def plot_importances(
    importance_results: Union[List["PermutationImportanceResults"], "PermutationImportanceResults"],
    labels: List[str],
    variables: List[str],
    bootstrapped:bool = True,
    swarmplot:bool = True,
    swarm_size:int = 3,
    figsize=None
):
    """
    Plots variable importances in the form of a swarmplot or a violinplot.

    Given a list of PermutationImportanceResults instances, this function plots a swarmplot (or violinplot) of each instance, displaying the importance of each variable.

    :param importance_results: List of importances dictionaries.
    :param labels: List for the subplot names.
    :param variables: List of the variable names.
    :param bootstrapped: If importance results come from a bootstrapped study, whether or not to use the importance bootstrapped quantities.
    :param swarmplot: Whether to plot a swarmplot or a violinplot.

    :return: A handle to a matplotlib figure and a list of matplotlib axes.
    """

    if isinstance(importance_results, PermutationImportanceResults):
        importances_list = [importance_results]
    else:
        importances_list = importance_results

    nlabels = len(importances_list)
    num_rows = int(np.ceil(nlabels / 3))
    fig, axes = plt.subplots(num_rows, 3, figsize=(4 * len(variables), 7 * num_rows) if figsize is None else figsize)

    variables = np.array(variables)   

    for i, ax in enumerate(axes.flatten()):
        if i >= nlabels:
            fig.delaxes(ax)
            continue

        pi_results = importances_list[i]
        importances_mean = pi_results.importances_bootstrapped if (bootstrapped and pi_results.bootstrapped) else pi_results.importances_mean

        ordered_importances = np.flip(np.argsort(importances_mean))
        swarms = pd.DataFrame(data=pi_results.importances[ordered_importances, :].T, columns=variables[ordered_importances])

        title_suffix = " - (baseline %s=%.4E)" % (pi_results.metric_name, pi_results.baseline_score)
        ax.set_title(labels[i] + title_suffix)
        plot_colors = itertools.islice(itertools.cycle(mpl.cm.get_cmap("tab10").colors[:2]), len(swarms.columns))
        if swarmplot:
            ax.plot(
                variables[ordered_importances],
                importances_mean[ordered_importances],
                "X-k",
                alpha=0.5,
                label="Mean importance",
            )
            sns.swarmplot(data=swarms, size=1, ax=ax, palette=plot_colors, label="Importance per permutation")
        else:
            sns.violinplot(data=swarms, ax=ax, palette=plot_colors)
        ax.tick_params(rotation=60)

        handles, labels = ax.get_legend_handles_labels()
        if swarmplot:
            ax.legend(handles[:2], labels[:2])

        ax.set_xlabel("Variable")
        ylabel_suffix = " (increment from baseline %s)"%(pi_results.metric_name)
        ax.set_ylabel("Importance" + ylabel_suffix)

        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        ax.grid(True)
        ax.grid(True, which="minor", alpha=0.2)
        ax.set_axisbelow(True)

    fig.tight_layout(pad=2)
    return fig, axes


def plot_importances_stackedbars(
    importance_results: Union[List["PermutationImportanceResults"], "PermutationImportanceResults"],
    labels: List[str],
    variables: List[str],
    scale="absolute",
    bootstrapped:bool = True
):
    """
    Plot a stacked barplot of variable importances.

    Given a list of PermutationImportanceResults instances, this function plots a stacked barplot of each instance, displaying the importance of each variable.

    :param importance_results: List of PermutationImportanceResults instances.
    :param labels: List for the subplot names.
    :param variables: List of the variable names.
    :param scale: Accepted values are 'absolute', 'relative', and 'percentage'. By default is set to 'absolute'.
        - 'absolute' : The height of each stacked bar is the mean importance of the variable
        - 'relative' : The height of each stacked bar is importance / abs(baseline_score)
        - 'percentage' : The height of each stacked bar is importance / sum(importances) * 100
    :param bootstrapped: If importance results come from a bootstrapped study, whether or not to use the importance bootstrapped quantities.
    :return: A handle to a matplotlib figure and a matplotlib axis.
    """

    colors_plot = itertools.cycle(mpl.cm.get_cmap("tab20").colors)
    color_dict = {variables[i]: next(colors_plot) for i in range(len(variables))}

    legends = []
    legend_parameters = []
    fig, ax = plt.subplots(1, figsize=(16, 8))

    t = ax.text(-1, -1, "test")
    t.set_fontsize(mpl.rcParams["axes.labelsize"])
    tick_fontsize = t.get_fontsize()
    t.remove()

    if isinstance(importance_results, PermutationImportanceResults):
        importances_list = [importance_results]
    else:
        importances_list = importance_results

    for i in range(len(importances_list)):
        pi_results = importances_list[i]
        importances_mean = pi_results.importances_bootstrapped if (bootstrapped and pi_results.bootstrapped) else pi_results.importances_mean

        ordered_importances = np.flip(np.argsort(importances_mean))

        stacked_bars = []
        legend_labels = []
        plotted_importances = []

        for j in range(len(ordered_importances)):
            var_name = variables[ordered_importances[j]]
            if scale == "absolute":
                importance = importances_mean[ordered_importances[j]]
            elif scale == "relative":
                importance = importances_mean[ordered_importances[j]] / abs(pi_results.baseline_score)
            elif scale == "percentage":
                importance = importances_mean[ordered_importances[j]] / sum(importances_mean) * 100

            if scale == "percentage":
                legend_labels.append(var_name + " (%.2f %%)" % importance)
            else:
                legend_labels.append(var_name + " (%.2f)" % importance)

            label_suffix = "\n(baseline %s=%.4E)" % (pi_results.metric_name, pi_results.baseline_score)
            stacked_bars.append(
                ax.bar(
                    labels[i]+label_suffix,
                    importance,
                    label=var_name + " (%.2f)" % importance,
                    color=color_dict[var_name],
                    bottom=sum(plotted_importances),
                )
            )
            plotted_importances.append(importance)

        legend_parameters.append([stacked_bars, legend_labels])

    ymin, ymax = ax.get_ylim()
    ymin_px, ymax_px = ax.get_window_extent().get_points()[:, 1]
    yscale = (ymax - ymin) / (ymax_px - ymin_px)

    ax.legend()

    ax.set_title("Importances (" + scale + ")")
    for legend in legends:
        ax.add_artist(legend)

    ylabel_suffix = " (increment from baseline %s)"%pi_results.metric_name
    ax.set_ylabel("Importance" + ylabel_suffix)

    return fig, ax


class SHAPStudy:
    """
    ## **SHAP Study**
    Class for handling SHAP values

    #### Inputs
    ###### shap_values
    Shapley values obtained with a SHAP explainer
    ###### indices
    Indices of the Pandas DataFrame used for the explanation
    ###### name
    Name of the study
    ###### description
    Optional description of the study
    """

    def __init__(self, base_values: np.ndarray, shap_values: np.ndarray, indices: np.ndarray, name, description: str = ""):
        """
        Initialize a SHAPStudy instance.

        :param base_values: The base values for the study.
        :param shap_values: The Shapley values obtained from a SHAP explainer.
        :param indices: Indices of the Pandas DataFrame used for the explanation.
        :param name: Name of the study.
        :param description: Optional description of the study.
        """
        self.shap_values = shap_values
        self.indices = indices
        self.base_values = base_values

        self.name = name
        self.description = description

    @classmethod
    def load_study(cls, filepath, name):
        """
        Load a previously computed study from a JSON file.

        :param filepath: Path to the JSON file.
        :param name: Name of the Study to load from the file.
        :return: Loaded SHAPStudy instance.
        """
        with open(filepath, "r") as file:
            loaded_dict = json.load(file)

        indices = np.array(loaded_dict[name]["indices"])
        shapley_values = np.array(loaded_dict[name]["shapley_values"])
        base_values = np.array(loaded_dict[name]["base_values"])

        return SHAPStudy(
            base_values,
            shapley_values,
            indices,
            name=loaded_dict[name]["name"],
            description=loaded_dict[name]["description"],
        )

    @classmethod
    def save_study(
        cls,
        studies: Union["SHAPStudy", List["SHAPStudy"]],
        filepath: str,
        names: Union[str, List[str]] = None,
        descriptions: Union[str, List[str]] = None,
    ):
        """
        Save a study to a JSON file.

        :param studies: SHAP Study (or list of studies) to be saved.
        :param filepath: Path to JSON file.
        :param names: Optional name of the Study (or list of names). If set to None, the name of the Study object will be used.
        :param descriptions: Optional description of the Study (or list of descriptions). If set to None, the description of the Study object will be used.
        """

        if isinstance(studies, list):
            if names is not None:
                if not isinstance(names, list):
                    raise ValueError("If a list of studies to save is given, 'names' must be a list of the names")
                if len(names) != len(studies):
                    raise ValueError("'studies' and 'names' must be of the same length")
            if descriptions is not None:
                if not isinstance(descriptions, list):
                    raise ValueError("If a list of studies to save is given, 'descriptions' must be a list of the descriptions")
                if len(descriptions) != len(studies):
                    raise ValueError("'studies' and 'descriptions' must be of the same length")

        else:
            studies = [studies]
            if names is not None:
                if not isinstance(names, str):
                    raise ValueError("'names' must be a string")
                names = [names]
            if descriptions is not None:
                if not isinstance(descriptions, str):
                    raise ValueError("'descriptions' must be a string")
                descriptions = [descriptions]

        if os.path.isfile(filepath):
            with open(filepath, "r") as file:
                dict_to_save = json.load(file)
        else:
            dict_to_save = {}

        for i, study in enumerate(studies):
            if names is not None:
                name_ = names[i]
            else:
                name_ = study.name

            if descriptions is not None:
                description_ = descriptions[i]
            else:
                description_ = study.description

            data_dict = {}
            data_dict["name"] = name_
            data_dict["description"] = description_

            data_dict["base_values"] = study.base_values.tolist()
            data_dict["indices"] = study.indices.tolist()
            data_dict["shapley_values"] = study.shap_values.tolist()

            dict_to_save[name_] = data_dict

        with open(filepath, "w") as file:
            json.dump(dict_to_save, file)

    @classmethod
    def combine_studies(cls, studies: Union[List["SHAPStudy"], List[str]], description: str = "", filepath: str = None):
        """
        Combine multiple SHAP Studies into a single study.

        :param studies: List of studies to combine.
        :param description: Optional description of the new study.
        :param filepath: If 'studies' is set to a list of names, filepath is the path to the JSON file containing those studies.
        :return: Combined SHAPStudy instance.
        """
        if isinstance(studies[0], str):
            if filepath is None:
                raise ValueError(
                    "If the studies are given as a list of names, the JSON file containing the studies must be specified by the 'filepath' argument"
                )
            studies_ = []
            for i in range(len(studies)):
                studies_.append(SHAPStudy.load_study(filepath, studies[i]))
        else:
            studies_ = studies

        n_variables = studies_[0].shap_values.shape[1]
        for i in range(len(studies_)):
            if studies_[i].shap_values.shape[1] != n_variables:
                raise ValueError("Every study must have the same number of variables")

        base_values = sum([study.base_values for study in studies_]) / len(studies_)
        shap_values = np.concatenate([study.shap_values for study in studies_], axis=0)
        indices = np.concatenate([study.indices for study in studies_], axis=0)

        return SHAPStudy(base_values, shap_values, indices, description)

    @classmethod
    def create_substudy(cls, study: "SHAPStudy", new_indices: List[int], name: str, description: str = ""):
        """
        Creates a substudy from an existing study and a subset of its indices.

        :param study: SHAP Study to subdivide.
        :param new_indices: Desired indices of the substudy.
        :param name: Name of the substudy.
        :param description: Description of the new substudy.
        :return: Substudy as a SHAPStudy instance.
        """
        _, masked_indices, _ = np.intersect1d(study.indices, new_indices, return_indices=True)
        new_shap_values = study.shap_values[masked_indices, :]
        new_study = SHAPStudy(
            base_values=study.base_values,
            shap_values=new_shap_values,
            indices=study.indices[masked_indices],
            description=description,
            name=name,
        )
        return new_study

    @classmethod
    def multiple_dependence_plots(
        cls,
        studies: List["SHAPStudy"],
        feature_values: Union[List[pd.DataFrame], pd.DataFrame],
        plotted_features: List[str] = None,
        filter_indices: List[np.ndarray] = None,
        labels: List[str] = None,
        x_jitter: float = 0,
        alpha: Union[float, np.ndarray] = 1,
        s: float = 10,
        y_scale_equal: bool = False,
        marginal_plots: str = "hist",
        nbins: int = 20,
        cmap: str = "coolwarm",
        marker: Union[List[str], str] = None,
        interaction_values: Union[List[np.ndarray], List[List[np.ndarray]]] = None,
        upper_interaction_limit: float = None,
        lower_interaction_limit: float = None,
        nWorkers: int = 1,
        figHsize: float = 24,
        figAspectRatio: float = 5,
        clabel: str = None,
        title: str = None,
        tight_layout: bool = True,
    ) -> mpl.figure.Figure:
        """
        Plots dependence plots for several studies. These studies must be elaborated from the same variables.

        :param studies: Studies to plot.
        :param feature_values: Variable values from which the studies were made.
        :param plotted_features: Names of the variables to be plotted. If set to None, all features from the feature_values variable will be plotted.
        :param labels: Legend labels for each study.
        :return: Matplotlib Figure.
        """

        if isinstance(feature_values, pd.DataFrame):
            feature_values = [feature_values] * len(studies)

        if plotted_features is None:
            plotted_features = feature_values[0].columns

        if labels is None:
            labels = ["_nolegend_"] * len(studies)

        if marginal_plots == "hist":
            show_hist = True
            show_kde = False
        elif marginal_plots == "kde":
            show_hist = False
            show_kde = True
        elif marginal_plots == "reg":
            show_hist = True
            show_kde = True
        elif marginal_plots is None:
            show_kde = False
            show_hist = False
        else:
            raise ValueError("marginal_plots addmited values are: 'hist', 'kde', 'reg' and None")

        nPlots = len(plotted_features)
        if interaction_values is None:
            cmap_ = cm.get_cmap("tab10")
            colors_plot = cmap_(range(len(studies)))
            cmap = None
        else:
            max_interaction = -np.inf
            min_interaction = np.inf

        indices = [study.indices for study in studies]

        if y_scale_equal:
            shap_min = min(min([[min(studies[j].shap_values[:, k]) for j in range(len(studies))] for k in range(nPlots)]))
            shap_max = max(max([[max(studies[j].shap_values[:, k]) for j in range(len(studies))] for k in range(nPlots)]))

            y_min = shap_min - 0.1 * (shap_max - shap_min)
            y_max = shap_max + 0.1 * (shap_max - shap_min)

            y_range = [y_min, y_max]

        def func(i, axes: plt.Axes):
            filter_indices_ = []
            if filter_indices is not None:
                if any([filter_indices[i][0].size > 1 for i in range(len(filter_indices))]):
                    for j in range(len(indices)):
                        filter_indices_.append(np.in1d(indices[j], filter_indices[i][j]))
                else:
                    for j in range(len(indices)):
                        filter_indices_.append(np.in1d(indices[j], filter_indices[j]))

            else:
                for j in range(len(indices)):
                    filter_indices_.append(np.ones(indices[j].shape, dtype=np.bool))

            if interaction_values is not None:
                if interaction_values[0][0].size > 1:
                    interaction_values_ = interaction_values[i]
                else:
                    interaction_values_ = interaction_values

                max_interaction = -np.inf
                min_interaction = np.inf

                for j in range(len(interaction_values_)):
                    if upper_interaction_limit is not None:
                        interaction_values_[j][interaction_values_[j] > upper_interaction_limit] = upper_interaction_limit
                    if lower_interaction_limit is not None:
                        interaction_values_[j][interaction_values_[j] < lower_interaction_limit] = lower_interaction_limit

                    if interaction_values_[j].max() > max_interaction:
                        max_interaction = interaction_values_[j].max()
                    if interaction_values_[j].min() < min_interaction:
                        min_interaction = interaction_values_[j].min()

            feature_idx = feature_values[0].columns.get_loc(plotted_features[i])
            features = np.concatenate([feature_values[j].loc[indices[j][filter_indices_[j]], :] for j in range(len(studies))])
            feature_range = [features[:, feature_idx].min(), features[:, feature_idx].max()]

            shap_value_range = np.concatenate([studies[j].shap_values[filter_indices_[j]][:, feature_idx] for j in range(len(studies))])
            shap_value_range = [shap_value_range.min(), shap_value_range.max()]

            for j in range(len(studies)):
                shap_values = studies[j].shap_values[filter_indices_[j]]
                features = feature_values[j].loc[indices[j][filter_indices_[j]], :]

                str_feature = any([isinstance(features.iloc[k, feature_idx], str) for k in range(features.shape[0])])

                x_jitter_ = (
                    st.norm.rvs(size=features.iloc[:, feature_idx].shape, loc=0, scale=x_jitter)
                    * (features.iloc[:, feature_idx].max() - features.iloc[:, feature_idx].min())
                    if not str_feature
                    else ""
                )

                if marker is not None:
                    if isinstance(marker, list):
                        marker_ = marker[j]
                    else:
                        marker_ = marker
                else:
                    marker_ = marker

                if interaction_values is None:
                    color_kwargs = {"color": colors_plot[j]}

                else:
                    color_kwargs = {
                        "c": interaction_values_[j].flatten()[filter_indices_[j]],
                        "vmin": min_interaction,
                        "vmax": max_interaction,
                    }

                if isinstance(alpha, (float, int)):
                    alpha_ = alpha
                else:
                    alpha_ = alpha[j]

                axes.scatter(
                    features.iloc[:, feature_idx] + x_jitter_,
                    shap_values[:, feature_idx],
                    cmap=cmap,
                    s=s,
                    label=labels[j],
                    alpha=alpha_,
                    marker=marker_,
                    **color_kwargs
                )

                if interaction_values is None and marginal_plots is not None:
                    # First study plotted -> create the inset axes
                    if j == 0:
                        lat_ax = axes.inset_axes([1, 0, 0.2, 1], sharey=axes)
                        top_ax = axes.inset_axes([0, 1, 1, 0.2], sharex=axes)

                    if len(features.iloc[:, feature_idx]) > 0:
                        if show_hist:
                            weigths = (
                                np.ones_like(features.iloc[:, feature_idx]) / len(features.iloc[:, feature_idx]) if not show_kde else None
                            )
                            numerator = features.iloc[:, feature_idx].max() - features.iloc[:, feature_idx].min()
                            denominator = feature_range[1] - feature_range[0]
                            if numerator == 0 or denominator == 0:
                                nbins_ = 1
                            else:
                                nbins_ = int(nbins * numerator / denominator)
                            nbins_ = max(1, nbins_)

                            top_ax.hist(
                                features.iloc[:, feature_idx],
                                bins=nbins_,
                                weights=weigths,
                                alpha=0.5 * alpha_,
                                linewidth=0.2,
                                edgecolor=np.array(colors_plot[j]).reshape(1,-1),
                                color=colors_plot[j],
                                density=show_kde,
                            )

                            weigths = np.ones_like(shap_values[:, feature_idx]) / len(shap_values[:, feature_idx]) if not show_kde else None
                            # nbins_ = nbins * (shap_values[:, feature_idx].max() - shap_values[:, feature_idx].min())
                            # nbins_ /= (y_range[1] - y_range[0]) if y_scale_equal else (shap_value_range[1] - shap_value_range[0])
                            numerator = shap_values[:, feature_idx].max() - shap_values[:, feature_idx].min()
                            denominator = (y_range[1] - y_range[0]) if y_scale_equal else (shap_value_range[1] - shap_value_range[0])
                            if numerator == 0 or denominator == 0:
                                nbins_ = 1
                            else:
                                nbins_ = int(nbins * numerator / denominator)
                            nbins_ = max(1, int(nbins_))

                            lat_ax.hist(
                                shap_values[:, feature_idx],
                                bins=nbins_,
                                weights=weigths,
                                alpha=0.5 * alpha_,
                                linewidth=0.2,
                                edgecolor=np.array(colors_plot[j]).reshape(1,-1),
                                color=colors_plot[j],
                                density=show_kde,
                                orientation="horizontal",
                            )
                        if show_kde:
                            kernel = st.gaussian_kde(np.array(features.iloc[:, feature_idx], dtype=np.float).flatten())
                            xplot = np.linspace(min(features.iloc[:, feature_idx]), max(features.iloc[:, feature_idx]), 100)
                            top_ax.plot(xplot, np.maximum(kernel(xplot), 0), alpha=alpha_, color=colors_plot[j])

                            kernel = st.gaussian_kde(np.array(shap_values[:, feature_idx], dtype=np.float).flatten())
                            xplot = np.linspace(min(shap_values[:, feature_idx]), max(shap_values[:, feature_idx]), 100)
                            lat_ax.plot(np.maximum(kernel(xplot), 0), xplot, alpha=alpha_, color=colors_plot[j])

                if interaction_values is None and marginal_plots is not None:
                    lat_ax.axis("off")
                    top_ax.axis("off")
                axes.autoscale()

            if y_scale_equal:
                axes.set_ylim(y_range)

            axes.set_xlabel(features.columns[feature_idx])
            axes.set_ylabel("SHAP value for\n" + str(features.columns[feature_idx]))

            axes.axhline(linestyle="--", color="k", zorder=-1)
            axes.legend()

            if interaction_values is None:
                return axes, None

            else:
                norm = colors.Normalize(vmin=min_interaction, vmax=max_interaction)
                mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

                return axes, mappable

        colorbar_kwargs = {}
        if upper_interaction_limit is not None and lower_interaction_limit is not None:
            colorbar_kwargs["extend"] = "both"
        elif upper_interaction_limit is not None:
            colorbar_kwargs["extend"] = "max"
        elif lower_interaction_limit is not None:
            colorbar_kwargs["extend"] = "min"

        fig = multiplePlots(
            nPlots,
            func,
            nWorkers,
            figHsize=figHsize,
            figAspectRatio=figAspectRatio,
            xlabel=None,
            ylabel=None,
            oneColorbar=False,
            tight_layout=tight_layout,
            clabel=clabel,
            colorbarKwargs=colorbar_kwargs,
        )

        if title is not None:
            fig.suptitle(title)

        return fig

    @classmethod
    def group_dependence_plot(
        cls,
        study: "SHAPStudy",
        feature_values: pd.DataFrame,
        feature_groups: List[List[str]],
        title: str = None,
        nWorkers: int = 1,
        figHsize: float = 24,
        figAspectRatio: float = 5,
    ) -> mpl.figure.Figure:
        """
        Plots Shapley values by groups of two variables. The plotted Shapley values will be the sum of the Shapley values obtained from those two variables.

        :param study: Study to be plotted.
        :param feature_values: Variable values from which the studies were made.
        :param feature_groups: List of pairs of variable names.
        :param title: Title of the plot.
        :return: Matplotlib Figure.
        """

        nPlots = len(feature_groups)

        def func(i, axes: plt.Axes):
            feature0 = feature_groups[i][0]
            feature1 = feature_groups[i][1]

            mask0 = feature_values.columns == feature0
            mask1 = feature_values.columns == feature1

            # The Shapley values of the grouped variables is approximately the sum of the Shapley values of each variable
            group_shap_values = (study.shap_values[:, mask0] + study.shap_values[:, mask1]).reshape(
                study.shap_values.shape[0],
            )

            scatter = axes.scatter(
                feature_values.loc[study.indices, feature0],
                feature_values.loc[study.indices, feature1],
                s=10,
                c=group_shap_values,
                cmap="coolwarm",
            )

            axes.tick_params("x", rotation=15)

            axes.set_xlabel(feature0)
            axes.set_ylabel(feature1)

            return axes, scatter

        fig = multiplePlots(
            nPlots,
            func,
            nWorkers,
            figHsize=figHsize,
            figAspectRatio=figAspectRatio,
            xlabel=None,
            ylabel=None,
            oneColorbar=False,
            tight_layout=False,
        )

        fig.suptitle(title)

        return fig

    @classmethod
    def dependence_plot_validation(
        cls,
        studies: List["SHAPStudy"],
        X: pd.DataFrame,
        model,
        grouped_vars: List[str],
        delta_thres: float = 0.1,
        eps_thres: float = 0.1,
        higherr_numsamples_thres: float = 0.1,
        q: float = 0.1,
        labels: List[str] = None,
        group_names: List[str] = None,
        ungrouped_vars: List[str] = None,
        always_plot_1stpc: bool = False,
        display_table: bool = False,
        display_pcs: bool = False,
        sparse_pca: bool = True,
        random_state: int = None,
    ):
        """
        Validate the dependence plot for a list of studies.

        :param studies: List of SHAPStudy instances.
        :param X: Pandas DataFrame.
        :param model: The model to validate against.
        :param grouped_vars: List of variable groups.
        :param delta_thres: Threshold for delta.
        :param eps_thres: Threshold for epsilon.
        :param higherr_numsamples_thres: Threshold for high error samples.
        :param q: Quantile value.
        :param labels: List of labels.
        :param group_names: List of group names.
        :param ungrouped_vars: List of ungrouped variables.
        :param always_plot_1stpc: Whether to always plot the first PC.
        :param display_table: Whether to display a table.
        :param display_pcs: Whether to display principal components.
        :param sparse_pca: Whether to use sparse PCA.
        :param random_state: Random state for PCA.
        """
        def gen_col_names(grouped_vars, k, ungrouped_vars):
            col_names = []
            for i, var in enumerate(grouped_vars):
                if hasattr(k, "__iter__"):
                    k_ = k[i]
                else:
                    k_ = k
                col_names += [var + " " + str(i + 1) for i in range(k_)]
            col_names += ungrouped_vars

            return col_names

        def weighted_rec_error(X, X_rec, y_hat, y_hat_rec, group_vars_cols, base_error="rmse"):
            X = np.array(X, dtype=np.float)
            X_rec = np.array(X_rec, dtype=np.float)
            y_hat = np.array(y_hat, dtype=np.float)
            y_hat_rec = np.array(y_hat_rec, dtype=np.float)

            delta = np.abs((y_hat_rec - y_hat) / y_hat)

            rec_error = X - X_rec

            min_rel_value = np.quantile(np.abs(X), q, axis=0)
            for col in range(X.shape[1]):
                denominator = np.where(np.abs(X[:, col]) > min_rel_value[col], np.abs(X[:, col]), np.nan)
                denominator[np.isnan(denominator)] = min_rel_value[col]

                rec_error[:, col] = np.abs(rec_error[:, col] / denominator)

            epsilon = np.empty([X.shape[0], len(group_vars_cols)])
            for i, group in enumerate(group_vars_cols):
                if base_error == "rmse":
                    epsilon[:, i] = np.sqrt(np.mean(np.square(rec_error[:, group]), axis=1))

                epsilon[:, i] = epsilon[:, i] * delta / delta_thres

            return epsilon

        if group_names is None:
            # TODO
            pass

        available_markers = ["o", "v", "^", "<", ">", "s", "P", "*", "X", "D"]
        marker_generator = itertools.cycle(available_markers)
        markers = [next(marker_generator) for _ in range(len(studies))]

        grouped_vars_flatten = [var for group in grouped_vars for var in group]
        group_vars_cols = [[grouped_vars_flatten.index(col) for col in group] for group in grouped_vars]

        # Vanilla model predictions
        yhat_model = model.predict(X)

        # Group Mapping
        group_mapping = GroupMapping()
        group_mapping.fit(X, grouped_vars)
        X_grmap = group_mapping.transform(X)
        X_grmap = pd.DataFrame(X_grmap, index=X.index, columns=group_names + ungrouped_vars)

        # Selection of number of PC for each group
        group_lengths = [len(group) for group in grouped_vars]
        for k in range(max(group_lengths)):
            n_components = [k + 1 if k + 1 < length else length for length in group_lengths]
            col_names = gen_col_names(group_names, n_components, ungrouped_vars)

            # Group PCA
            gr_pca = models.pca.GroupedPCA(n_components=n_components, numeric_norm="minMax")
            gr_pca.fit(X, grouped_vars)
            X_grpca = gr_pca.transform(X)
            X_grpca = pd.DataFrame(data=X_grpca, index=X.index, columns=col_names)

            pipe_pca = PCA_Pipeline(gr_pca, model)

            X_rec = gr_pca.inverse_transform(gr_pca.transform(X))
            X_rec = pd.DataFrame(data=X_rec, index=X.index, columns=X.columns)

            # PCA Pipeline predictions
            yhat_pca = pipe_pca.predict(X_grpca)

            proportion_higherr_samples = (
                sum(np.any(weighted_rec_error(X[grouped_vars_flatten], X_rec[grouped_vars_flatten], yhat_model, yhat_pca, group_vars_cols) > eps_thres, axis=1)) / X.shape[0]
            )

            if proportion_higherr_samples < higherr_numsamples_thres:
                break

        #print(n_components, proportion_higherr_samples)

        masked_indices_list = [study.indices for study in studies]*2

        alphas = [alpha for sublist in [[elem]*len(studies) for elem in [0.05, 1]] for alpha in sublist]

        for k in range(max(n_components)):
            k_components = [k + 1 if k + 1 < n else n for n in n_components]
            print("Group PCA with n_components=",k_components)

            col_names = gen_col_names(group_names, k_components, ungrouped_vars)

            # Group PCA
            gr_pca = models.pca.GroupedPCA(
                n_components=k_components, numeric_norm="minMax", sparse=sparse_pca, transformer_kwargs={"random_state": random_state}
            )
            gr_pca.fit(X, grouped_vars)
            X_grpca = gr_pca.transform(X)
            X_grpca = pd.DataFrame(data=X_grpca, index=X.index, columns=col_names)

            X_grmap_enc = group_mapping.encode(X_grmap, mapped_values=X_grpca, return_df=True, columns=col_names, index=X_grmap.index, values_per_group=k+1)

            pipe_pca = PCA_Pipeline(gr_pca, model)

            X_rec = gr_pca.inverse_transform(gr_pca.transform(X))
            X_rec = pd.DataFrame(data=X_rec, index=X.index, columns=X.columns)

            # PCA Pipeline predictions
            yhat_pca = pipe_pca.predict(X_grpca)

            epsilon = weighted_rec_error(X[grouped_vars_flatten], X_rec[grouped_vars_flatten], yhat_model, yhat_pca, group_vars_cols)
            epsilon = pd.DataFrame(data=epsilon, index=X.index, columns=group_names)

            epsilon_list = []
            for i, gr_name in enumerate(group_names):
                aux_list = []
                for study in studies:
                    epsilon_study = epsilon.loc[study.indices, gr_name].to_numpy()
                    aux_list.append(epsilon_study)
                epsilon_list.append(aux_list)

            if always_plot_1stpc and k != 0:
                pca_pcplotted = []
                for i, n_pc in enumerate(k_components):
                    pca_pcplotted.append(sum(k_components[:i]))

                cls.multiple_dependence_plots(studies, X_grmap_enc.iloc[:, pca_pcplotted], labels=labels, plotted_features=X_grmap_enc.columns[pca_pcplotted], interaction_values=epsilon_list, cmap="cool", title="Rec. Error on Input", x_jitter=0.01, upper_interaction_limit=1, marker=markers, s=20, tight_layout=False)
                plt.show()

                cls.multiple_dependence_plots(studies*2, X_grmap_enc.iloc[:, pca_pcplotted], labels=labels*2, plotted_features=X_grmap_enc.columns[pca_pcplotted], filter_indices=masked_indices_list, marker=markers*2, s=20, tight_layout=True, alpha=alphas, y_scale_equal=True)
                plt.show()

            pca_pcplotted = []
            for i, n_pc in enumerate(k_components):
                pca_pcplotted.append(sum(k_components[:i]) + n_pc - 1)

            cls.multiple_dependence_plots(studies, X_grmap_enc.iloc[:, pca_pcplotted], labels=labels, plotted_features=X_grmap_enc.columns[pca_pcplotted], interaction_values=epsilon_list, cmap="cool", title="Rec. Error on Input", x_jitter=0.01, upper_interaction_limit=1, marker=markers, s=20, tight_layout=False)
            plt.show()

            cls.multiple_dependence_plots(studies*2, X_grmap_enc.iloc[:, pca_pcplotted], labels=labels*2, plotted_features=X_grmap_enc.columns[pca_pcplotted], filter_indices=masked_indices_list, marker=markers*2, s=20, tight_layout=True, alpha=alphas, y_scale_equal=True)
            plt.show()

            masked_indices_list = []
            table = pd.DataFrame()
            for i, group in enumerate(grouped_vars):
                aux_list = []
                for j in range(len(studies)):
                    mask = epsilon_list[i][j] > eps_thres
                    masked_indices = studies[j].indices[mask]
                    aux_list.append(masked_indices)

                    table.loc[labels[j], group_names[i]] = masked_indices.size / studies[j].indices.size

                masked_indices_list.append([study.indices for study in studies] + aux_list)

            if display_table:
                print("Porportion of high reconstruction error samples")
                print(table)
                print()

            if display_pcs:
                for i, pca in enumerate(gr_pca.pcas_):
                    print("Group PCA for ", group_names[i])
                    table = pd.DataFrame()
                    for j, pc in enumerate(pca.components_):
                        for m, pc_elem in enumerate(pc):
                            table.loc["PC"+str(j+1), grouped_vars[i][m]] = pc_elem
                    print(table)
                    print()


class PCA_Pipeline:
    """
    A pipeline that combines grouped PCA transformation with a predictive model for making predictions.

    :param grouped_pca: The grouped PCA model used for data transformation.
    :param model: The predictive model, such as regression or classification, to make predictions.

    Methods:
    :method predict: Predict using the combined PCA transformation and the predictive model.
    """
    def __init__(self, grouped_pca: models.pca.GroupedPCA, model):
        """
        Initialize a PCA_Pipeline.

        :param grouped_pca: The grouped PCA model used for data transformation.
        :param model: The predictive model, such as regression or classification, to make predictions.
        """
        self.grouped_pca = grouped_pca
        self.model = model

    def predict(self, X):
        """
        Make predictions on input data using the PCA transformation and the predictive model.

        :param X: Input data to make predictions on.
        :return: Predicted values.
        """
        X = np.array(X)
        X_inv = self.grouped_pca.inverse_transform(X)
        try:
            X_inv = X_inv.astype(np.float64)
        except:
            pass

        with HiddenPrints():
            y_hat = self.model.predict(X_inv)

        return y_hat


class Mapping_Pipeline:
    """
    A pipeline for mapping transformations using training and test mappings and making predictions with a model.

    :param mapping_train: The mapping for training data.
    :param mapping_test: The mapping for test data.
    :param model: The predictive model, such as regression or classification, to make predictions.

    Methods:
    :method predict: Predict using the mapping transformations and the predictive model.
    """
    def __init__(self, mapping_train: "GroupMapping", mapping_test: "GroupMapping", model):
        """
        Initialize a Mapping_Pipeline.

        :param mapping_train: The mapping for training data.
        :param mapping_test: The mapping for test data.
        :param model: The predictive model, such as regression or classification, to make predictions.
        """
        self.mapping_tr = mapping_train
        self.mapping_test = mapping_test
        self.model = model

    def predict(self, X):
        """
        Make predictions using the mapping transformations and the predictive model.

        :param X: Input data to make predictions on.
        :return: Predicted values.
        """
        X = np.array(X)
        X_inv_test = self.mapping_test.inverse_transform(X)
        X_inv_train = self.mapping_tr.inverse_transform(X)
        X_inv = np.where(X_inv_test == None, X_inv_train, X_inv_test)
        try:
            X_inv = X_inv.astype(np.float64)
        except:
            pass

        with HiddenPrints():
            y_hat = self.model.predict(X_inv)

        return y_hat


class GroupMapping(BaseEstimator, TransformerMixin):
    """
    A class for grouping and transforming data for machine learning.

    Methods:
    :method fit: Fit the GroupMapping to the input data with specified variable groups.
    :method transform: Transform the input data using the fitted GroupMapping.
    :method fit_transform: Fit the GroupMapping to the input data and transform it.
    :method inverse_transform: Inverse transform the input data using the fitted GroupMapping.
    :method encode: Encode the input data based on specified options.

    Attributes:
    :attribute variable_groups_: The variable groups used for grouping and transforming data.
    """

    __GLOBAL_STARTING_INDEX = 0

    def __init__(self):
        """
        Initialize a GroupMapping.
        """
        pass

    def fit(self, X, variable_groups: Union[List[int], List[str]]):
        """
        Fit the GroupMapping to the input data with specified variable groups.

        :param X: Input data to fit the GroupMapping.
        :param variable_groups: List of variable groups to use for grouping and transforming data.
        """
        if isinstance(X, pd.DataFrame) and isinstance(variable_groups[0][0], str):
            self.variable_groups_ = [[X.columns.get_loc(col) for col in group] for group in variable_groups]
            flattened_groups = [col for group in variable_groups for col in group]
            unaltered_columns = [col for col in X.columns if col not in flattened_groups]
            self.unaltered_columns_ = [X.columns.get_loc(col) for col in unaltered_columns]
        else:
            self.variable_groups_ = variable_groups
            flattened_groups = [col for group in variable_groups for col in group]
            self.unaltered_columns_ = [col for col in range(X.shape[1]) if col not in flattened_groups]

        self.starting_idx_ = GroupMapping.__GLOBAL_STARTING_INDEX
        GroupMapping.__GLOBAL_STARTING_INDEX += X.shape[0]
        self.X_ = np.array(X)
        self.fitted_ = True

    def transform(self, X) -> np.array:
        """
        Transform the input data using the fitted GroupMapping.

        :param X: Input data to transform.
        :return: Transformed data.
        """
        if not hasattr(self, "fitted_"):
            raise ValueError("This estimator has not been fitted!")

        X = np.array(X)
        transformed_X = np.empty([X.shape[0], len(self.variable_groups_)])

        for i, group in enumerate(self.variable_groups_):
            transformed_X[:, i] = np.arange(X.shape[0]) + self.starting_idx_

        output_X = np.concatenate([transformed_X, X[:, self.unaltered_columns_]], axis=1)

        return output_X

    def fit_transform(self, X, variable_groups: Union[List[int], List[str]]) -> np.array:
        """
        Fit the GroupMapping to the input data and transform it.

        :param X: Input data to fit and transform.
        :param variable_groups: List of variable groups to use for grouping and transforming data.
        :return: Transformed data.
        """
        self.fit(X, variable_groups)
        return self.transform(X)

    def inverse_transform(self, X) -> np.array:
        """
        Inverse transform the input data using the fitted GroupMapping.

        :param X: Input data to inverse transform.
        :return: Inverse transformed data.
        """
        X = np.array(X)
        inverse_X = np.empty([X.shape[0], self.X_.shape[1]], dtype=object)
        for i, group in enumerate(self.variable_groups_):
            mask = np.in1d(X[:, i], np.arange(self.X_.shape[0]) + self.starting_idx_)
            for col in group:
                indices = X[mask, i].astype(int) - self.starting_idx_
                if indices.size:
                    inverse_X[mask, col] = self.X_[indices, col]

        if len(self.unaltered_columns_):
            inverse_X[:, self.unaltered_columns_] = X[:, -len(self.unaltered_columns_) :]

        return inverse_X

    def encode(self, X, func=None, mapped_values=None, return_df=False, columns=None, index=None, values_per_group=1, **kwargs) -> np.array:
        """
        Encode the input data based on specified options.

        :param X: Input data to encode.
        :param func: A function to apply to the data.
        :param mapped_values: Predefined mapped values to use for encoding.
        :param return_df: Whether to return a DataFrame.
        :param columns: Column names for the DataFrame (required if return_df is True).
        :param index: Index for the DataFrame (required if return_df is True).
        :param values_per_group: Number of values per group to encode.
        :param kwargs: Additional keyword arguments for the encoding function.
        :return: Encoded data.
        """
        X = np.array(X)
        # encoded_X = np.empty(X.shape, dtype=object)
        encoded_X = np.empty([X.shape[0], len(self.variable_groups_) * values_per_group + len(self.unaltered_columns_)], dtype=object)

        if func is not None:
            for i, group in enumerate(self.variable_groups_):
                encoded_X[:, i] = func(self.X_[X[:, i].astype(int), :][:, group], **kwargs)
        elif mapped_values is not None:
            mapped_values = np.array(mapped_values)
            for i in range(len(self.variable_groups_) * values_per_group):
                encoded_X[:, i] = mapped_values[:, i]
        else:
            raise ValueError("At least one of the following arguments must be set: 'func', 'mapped_values'")

        # for i, group in enumerate(self.variable_groups_):
        #     if func is not None:
        #         encoded_X[:, i] = func(self.X_[X[:, i].astype(int), :][:, group], **kwargs)
        #     elif mapped_values is not None:
        #         mapped_values = np.array(mapped_values)
        #         encoded_X[:, i] = mapped_values[:, i]
        #     else:
        #         raise ValueError("At least one of the following arguments must be set: 'func', 'mapped_values'")

        encoded_X[:, i + 1 :] = X[:, -len(self.unaltered_columns_) :]

        if return_df:
            if columns is None or index is None:
                raise ValueError("When return_df is set to True, the parameters 'columns' and 'index' must be set")
            encoded_X = pd.DataFrame(data=encoded_X, columns=columns, index=index)

        return encoded_X

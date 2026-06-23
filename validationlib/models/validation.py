from typing import Union, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..plots.common import multiplePlots
from ..tests.interval import percentileBootstrap

def training_curves_plot(
    training_metrics: dict,
    validation_metrics: dict,
    plot_by_epoch: bool = True,
    n_iters_per_epoch: int = None,
    xlabel: str = None,
    title: str = None,
    ylogscale: Union[bool, list[bool]] = True,
    figHsize: float = 7,
    figAspectRatio: float = 1.5,
):
    """
    Plot training and validation curves for a given set of metrics.

    :param training_metrics: Dictionary containing the training metrics. The keys are the names of the metrics and the values are lists containing the values of the metrics.
    :param validation_metrics: Dictionary containing the validation metrics. The keys are the names of the metrics and the values are lists containing the values of the metrics.
    :param plot_by_epoch: If True, the x-axis will be the epoch number. If False, the x-axis will be the iteration number.
    :param n_iters_per_epoch: Number of iterations per epoch. If None, and if training metrics and validation metrics have different legth, it is assumed that the training metrics are provided by iteration and the validation metrics by epoch.
    :param xlabel: Label for the x-axis. If None, the x-axis will be labeled as "Epoch" if plot_by_epoch is True, and "Iteration" if plot_by_epoch is False.
    :param title: Title of the plot.
    :param ylogscale: If True, the y-axis will be in log scale. If False, the y-axis will be in linear scale. If a list of booleans is provided, each element will determine if the corresponding metric will be plotted in log scale.
    :param figHsize: Height of the figure.
    :param figAspectRatio: Aspect ratio of the figure.

    :return: Figure containing the training and validation curves.
    """
    training_metrics_keys = list(training_metrics.keys())
    validation_metrics_keys = list(validation_metrics.keys())

    metrics_keys_set = list(set(training_metrics_keys + validation_metrics_keys))
    n_plots = len(metrics_keys_set)

    def func(i, axs: plt.Axes):
        training_metric = training_metrics.get(metrics_keys_set[i], None)
        validation_metric = validation_metrics.get(metrics_keys_set[i], None)

        if xlabel is not None:
            xlabel_ = xlabel
        else:
            xlabel_ = "Epoch" if plot_by_epoch else "Iteration" 

        # Check if the length of the training and validation metrics are the same
        if training_metric is not None and validation_metric is not None:
            if len(training_metric) == len(validation_metric):
                x_training = np.arange(1, len(training_metric) + 1)
                x_validation = x_training

            else:
                # If training and validation metrics have different lengths, training metric is assumed to be provided by iteration (batch) and validation metric by epoch
                x_training = np.arange(1, len(training_metric) + 1)
                if n_iters_per_epoch is None:
                    n_iters_per_epoch_ = len(training_metric) // len(validation_metric)
                else:
                    n_iters_per_epoch_ = n_iters_per_epoch

                if plot_by_epoch:
                    x_training = np.arange(1, len(training_metric)//n_iters_per_epoch_ + 1)
                    x_validation = np.arange(1, len(validation_metric) + 1)
                    training_metric = training_metric[::n_iters_per_epoch_]
                else:
                    x_validation = np.arange(n_iters_per_epoch_, len(training_metric) + 1, n_iters_per_epoch_)

        if training_metric is not None:
            axs.plot(x_training, training_metric, label='Training')
        if validation_metric is not None:
            axs.plot(x_validation, validation_metric, label='Validation')

        if isinstance(ylogscale, bool):
            axs.set_yscale('log' if ylogscale else 'linear')
        elif isinstance(ylogscale, list):
            axs.set_yscale('log' if ylogscale[i] else 'linear')

        axs.grid()
        axs.minorticks_on()
        axs.grid(which='minor', linestyle=':', linewidth='0.5')
        axs.set_axisbelow(True)

        # Automatic minor ticking for the x axis with integer numbers
        axs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axs.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        axs.set_xlabel(xlabel_)
        axs.set_title(metrics_keys_set[i])
        axs.legend()

        return axs, None

    fig = multiplePlots(
        nplots=n_plots,
        func=func,
        figHsize=figHsize,
        figAspectRatio=figAspectRatio,
        xlabel=None,
        ylabel=None,
        oneColorbar=False,
        tight_layout=False,
    )

    fig.suptitle(title)

    return fig


@dataclass
class LearningCurvesResults:
    """
    Dataclass containing the results of the learning curves analysis.

    :param used_training_proportion: Proportion of the training data used in each iteration.
    :param used_training_size: Size of the training data used in each iteration.
    :param train_scores: Training scores.
    :param val_scores: Validation scores.
    :param bootstrapped: If True, the results were bootstrapped.
    :param ci: Confidence interval.
    :param train_scores_ci: Confidence interval for the training scores mean.
    :param val_scores_ci: Confidence interval for the validation scores mean.
    :param train_scores_sim: Bootstrapped training scores.
    :param val_scores_sim: Bootstrapped validation scores.
    :param n_repetitions: Number of repetitions for the bootstrapping.
    :param n_sim: Number of simulations (resamples) for the bootstrapping.
    """
    used_training_proportion: np.ndarray
    used_training_size: np.ndarray
    train_scores: np.ndarray
    val_scores: np.ndarray
    bootstrapped: bool = False
    ci: float = None
    train_scores_ci: np.ndarray = None
    val_scores_ci: np.ndarray = None
    train_scores_sim: np.ndarray = None
    val_scores_sim: np.ndarray = None
    n_repetitions: int = None
    n_sim: int = None

def learning_curves(
    training_method: Callable,
    train_df: Union[pd.DataFrame, np.ndarray, pd.api.typing.DataFrameGroupBy],
    validation_df: Union[pd.DataFrame, np.ndarray, pd.api.typing.DataFrameGroupBy],
    input_variables: list,
    output_variables: list,
    used_training_data: list = None,
    bootstrap: bool = False,
    n_repetitions: int = 10,
    n_sim: int = 100,
    ci: float = 0.95,
    random_state: int = None
):
    """
    Compute learning curves (loss vs training size) for a given training method.

    :param training_method: Training method. It must be a function that receives the training data, the validation data, the input variables and the output variables, and returns the training and validation scores. Its signature must be: training_method(training_data, validation_data, input_variables, output_variables).
    :param train_df: Training data. It can be a pandas DataFrame, a numpy array or a DataFrameGroupBy object.
    :param validation_df: Validation data. It can be a pandas DataFrame, a numpy array or a DataFrameGroupBy object.
    :param input_variables: List of input variables.
    :param output_variables: List of output variables.
    :param used_training_data: List of integers or floats representing the number of training data used in each iteration. If integers, they represent the number of training points used. If floats, they represent the proportion of training data used.
    :param bootstrap: If True, the results will be bootstrapped.
    :param n_repetitions: Number of repetitions for the bootstrapping.
    :param n_sim: Number of simulations (resamples) for the bootstrapping.
    :param ci: Confidence interval (only when bootstrapping).
    :param random_state: Random state for the choice of the training data.

    :return: Results of the learning curves analysis.
    """
    train_scores = []
    val_scores = []

    if bootstrap:
        train_scores_ci = []
        val_scores_ci = []
        train_scores_sim = []
        val_scores_sim = []

    def __training_iteration(indices):
        if isinstance(train_df, pd.DataFrame):
            training_subset = train_df.iloc[indices]
            train_score, val_score = training_method(training_subset, validation_df, input_variables, output_variables)
        elif isinstance(train_df, np.ndarray):
            training_subset = train_df[indices]
            train_score, val_score = training_method(training_subset, validation_df, input_variables, output_variables)
        elif isinstance(train_df, pd.api.typing.DataFrameGroupBy):
            groups = list(train_df.groups.keys())
            chosen_groups = [groups[idx] for idx in indices]
            training_subset = pd.concat((train_df.get_group(group_name) for group_name in chosen_groups))
            train_score, val_score = training_method(training_subset, validation_df, input_variables, output_variables)
        else:
            raise ValueError("train_df must be a pandas DataFrame, a numpy array or a DataFrameGroupBy object")

        return train_score, val_score


    if used_training_data is None:
        used_training_data = (np.linspace(0.1, 1, 10)*len(train_df)).astype(int)
    elif any([isinstance(elem, float) for elem in used_training_data]):
        if not all([elem >= 0 or elem <= 1 for elem in used_training_data]):
            raise ValueError("If values of used_training_data are floats, they must be between 0 and 1")
        used_training_data = (np.array(used_training_data)*len(train_df)).astype(int)
    elif all([isinstance(elem, int) for elem in used_training_data]):
        if any([elem < 0 or elem > len(train_df) for elem in used_training_data]):
            raise ValueError("If values of used_training_data are integers, they must be between 0 and the length of the training data")
    else:
        raise ValueError("Values of used_training_data must be either floats (spanning from 0 to 1) or integers (spanning from 0 to the length of the training data).")
    
    used_training_data_size = np.array(used_training_data)
    used_training_data_proportion = np.array(used_training_data)/len(train_df)

    rng = np.random.default_rng(random_state)
    if bootstrap:
        index_permutation = np.vstack([rng.permutation(len(train_df)) for _ in range(n_repetitions)])
        qinf = (1-ci)/2
        qsup = 1 - qinf
    else:
        index_permutation = rng.permutation(len(train_df))

    for n in tqdm(used_training_data):
        if bootstrap:
            train_scores_sim_i = []
            val_scores_sim_i = []

            for i in range(n_repetitions):
                train_score, val_score = __training_iteration(index_permutation[i, :n])
                train_scores_sim_i.append(train_score)
                val_scores_sim_i.append(val_score)

            bootstrapper = percentileBootstrap(data=np.array(train_scores_sim_i), conf=ci, nsim=n_sim, random_state=random_state)
            train_score, train_score_ci = bootstrapper.compute(np.mean)
            # train_score, _= bootstrapper.compute(np.mean)
            # train_score_qinf, _ = bootstrapper.compute(np.quantile, q=qinf)
            # train_score_qsup, _ = bootstrapper.compute(np.quantile, q=qsup)
            # train_score_ci = [train_score_qinf, train_score_qsup]

            bootstrapper = percentileBootstrap(data=np.array(val_scores_sim_i), conf=ci, nsim=n_sim, random_state=random_state)
            val_score, val_score_ci = bootstrapper.compute(np.mean)
            # val_score, _= bootstrapper.compute(np.mean)
            # val_score_qinf, _ = bootstrapper.compute(np.quantile, q=qinf)
            # val_score_qsup, _ = bootstrapper.compute(np.quantile, q=qsup)
            # val_score_ci = [val_score_qinf, val_score_qsup]

            train_scores.append(train_score)
            val_scores.append(val_score)
            train_scores_ci.append(train_score_ci)
            val_scores_ci.append(val_score_ci)
            train_scores_sim.append(train_scores_sim_i)
            val_scores_sim.append(val_scores_sim_i)

        else:
            train_score, val_score = __training_iteration(index_permutation[:n])
            train_scores.append(train_score)
            val_scores.append(val_score)

    if bootstrap:
        results = LearningCurvesResults(
            used_training_proportion=np.array(used_training_data_proportion),
            used_training_size=np.array(used_training_data_size),
            train_scores=np.array(train_scores),
            val_scores=np.array(val_scores),
            bootstrapped=True,
            ci=ci,
            train_scores_ci=np.array(train_scores_ci),
            val_scores_ci=np.array(val_scores_ci),
            train_scores_sim=np.array(train_scores_sim),
            val_scores_sim=np.array(val_scores_sim),
            n_repetitions=n_repetitions,
            n_sim=n_sim
        )
    else:
        results = LearningCurvesResults(
            used_training_proportion=np.array(used_training_data_proportion),
            used_training_size=np.array(used_training_data_size),
            train_scores=np.array(train_scores),
            val_scores=np.array(val_scores)
        )

    return results

def learning_curves_plot(
    learning_curves_result: LearningCurvesResults,
    figsize=(10, 6),
    plot_training_size=True,
    scatter: bool = True,
):
    """
    Plot learning curves (loss vs training size).

    :param learning_curves_result: Results of the learning curves analysis. Obtained from the learning_curves function.
    :param figsize: Figure size.
    :param plot_training_size: If True, the x-axis will be the training size. If False, the x-axis will be the proportion of training data used.

    :return: Figure containing the learning curves.
    """  
    fig, axs = plt.subplots(figsize=figsize)
    x_data = learning_curves_result.used_training_size if plot_training_size else learning_curves_result.used_training_proportion
    axs.plot(x_data, learning_curves_result.train_scores, label="Train", color="tab:blue")
    axs.plot(x_data, learning_curves_result.val_scores, label="Validation", color="tab:orange")

    if scatter:
        axs.scatter(np.vstack([x_data]*learning_curves_result.n_repetitions).T, learning_curves_result.train_scores_sim, c="tab:blue", s=2)
        axs.scatter(np.vstack([x_data]*learning_curves_result.n_repetitions).T, learning_curves_result.val_scores_sim, c="tab:orange", s=2)

    if learning_curves_result.bootstrapped:
        axs.fill_between(x_data, learning_curves_result.train_scores_ci[:,0], learning_curves_result.train_scores_ci[:,1], alpha=0.3, label=f"Mean CI{learning_curves_result.ci*100} Train", facecolor='tab:blue')
        axs.fill_between(x_data, learning_curves_result.val_scores_ci[:,0], learning_curves_result.val_scores_ci[:,1], alpha=0.3, label=f"Mean CI{learning_curves_result.ci*100} Validation", facecolor='tab:orange')

    axs.grid()
    axs.set_axisbelow(True)
    if plot_training_size:
        if len(learning_curves_result.used_training_size) > 15:
            axs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        else:
            axs.set_xticks(learning_curves_result.used_training_size)

    axs.legend()
    axs.set_xlabel("Training size" if plot_training_size else "Kept training proportion")
    axs.set_ylabel("Loss")

    return fig
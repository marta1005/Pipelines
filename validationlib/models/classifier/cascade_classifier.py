'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''

import copy
import time
from typing import List, Tuple, Dict, Union

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.model_selection import train_test_split

from .confusion_matrix import (
    normalized_confusion_matrices,
    list_normalized_cms,
    plot_normalized_cms,
)


class CascadeLevel:
    """
    ## **Cascade level**
    Defines the parameters of a cascade level for a cascade classifier.

    #### Inputs
    ###### clf
    This is the cascade level's classifier. It must implement a sklearn-like API (with `fit` and `predict` or `predict_proba` methods).
    ###### threshold
    If the classifier is a soft classifier, this threshold is used to adjust the "confidence" with which the sample is classified as the majority class.
    ###### clf_kwargs
    Dictionary of keyword arguments for the cascade level's classifier.
    ###### predict_proba
    States whether a soft classifier will output a soft or hard classification. If the cascade level's classifier is a hard one, this parameter is ignored.
    """

    def __init__(
        self,
        clf: BaseEstimator,
        threshold: float = 0.5,
        clf_kwargs: dict = None,
        predict_proba: bool = True,
    ):
        self.clf = clf
        self.threshold = threshold
        self.clf_kwargs = clf_kwargs
        self.predict_proba = predict_proba


class CascadeClassifier(BaseEstimator):
    """
    ## **Cascade classifier**
    Defines a meta-classifier composed of several classifiers disposed in a cascade fashion.

    #### Inputs
    ###### cascade_levels
    Defines the structure of the cascade. Must be a list of CascadeLevel objects.
    ###### until_balanced
    If set to true the cascade training will continue until the unbalance in the dataset is gone. If the unbalance is flipped around (the majority class is now the minority class), extra negative samples will be collected to train the last level. By default is set to true.
    ###### early_stopping
    If set to true, the cascade will train all of the defined cascade levels in cascade_levels. By default is set to false.

    #### Additional attributes
    ###### train_history_
    Saves the history of the training in two separate lists, accesible with the following keys:
        - "train_size": Stores the size of the training dataset slice that get passed to each cascade level.
        - "train_cms": Stores the total confusion matrix obtained at each cascade level.
    ###### fit_time_
    Stores the training time in seconds of each cascade level.
    """

    def __init__(self, cascade_levels: List[CascadeLevel], until_balanced: bool = False, early_stopping: bool = True):
        self.cascade_levels = cascade_levels
        self.until_balanced = until_balanced
        self.early_stopping = early_stopping

        for level in self.cascade_levels:
            if level.clf_kwargs is not None:
                level.clf.set_params(**level.clf_kwargs)

    def fit(self, X, y):
        """
        ### **fit**
        Fits the cascade to a set of data.

        #### Inputs
        ###### X
        Contains the explanatory variables of the training data
        ###### y
        Contains the target values of the training data
        """
        X = np.array(X)
        y = np.array(y)

        # This numpy array contains the total prediction of the model and is used to keep track of the training confusion matrices.
        y_pred = np.empty(X.shape[0])

        # The indices array will store the indices of each training data slice that get passed to each cascade level
        indices = []
        indices.append(np.arange(X.shape[0]))

        self.train_history_ = {"train_size": [], "train_cms": []}

        counts = np.unique(y, return_counts=True)[1]
        self.train_history_["train_size"].append(np.append(counts, X.shape[0]))

        self.fit_time_ = []

        # Variables used to control whether perfect precision has been achieved.
        # Once the cascade level's precision reaches 1, it no longer makes sense to keep adding cascade levels, so the training is stopped.
        early_stop = False
        train_last_level = False
        k = 0

        for k in range(len(self.cascade_levels)):
            if train_last_level:
                level = self.cascade_levels[-1]
            else:
                level = self.cascade_levels[k]

            X_train = X[indices[-1], :]
            y_train = y[indices[-1]]

            t_start = time.time()
            level.clf.fit(X_train, y_train)
            self.fit_time_.append(time.time() - t_start)

            # If the cascade level's classifier has a predict_proba method (is a soft classifier) and the level's attribute predict_proba is set to True, a soft classification is made. In any other case, a hard classification is made.
            if callable(getattr(level.clf, "predict_proba", None)) and level.predict_proba:
                y_hat = np.where(level.clf.predict_proba(X_train)[:, 0] > level.threshold, 0, 1)
                y_pred[indices[-1]] = np.where(level.clf.predict_proba(X_train)[:, 0] > level.threshold, 0, 1)
            else:
                y_hat = level.clf.predict(X_train)
                y_pred[indices[-1]] = level.clf.predict(X_train)

            indices.append(indices[-1][y_hat == 1])

            count_matrix = np.unique(y[indices[-1]], return_counts=True)
            counts = count_matrix[1]

            # If len(counts) < 2 means that the cascade level has reached a precision of 1
            if len(counts) < 2:
                if 0 in count_matrix[0]:
                    counts = np.array([counts[0], 0])
                elif 1 in count_matrix[0]:
                    counts = np.array([0, counts[0]])

                early_stop = True

            self.train_history_["train_size"].append(np.append(counts, indices[-1].shape[0]))
            self.train_history_["train_cms"].append(confusion_matrix(y_true=y, y_pred=y_pred))

            if early_stop and self.early_stopping:
                break

            # If until_balanced is set to true, the cascade training will continue until the unbalance in the dataset is gone.
            # If the unbalance is flipped around (the majority class is now the minority class), extra negative samples will be
            # collected to train the last level.
            if self.until_balanced and counts[0] < counts[1] and level is not self.cascade_levels[-1]:
                # y_lastlevel = y[indices[-1]]
                # y_previouslevel = y[indices[-2]]
                new_possible_indices = indices[-2][np.isin(indices[-2], indices[-1], invert=True)]
                new_possible_indices = new_possible_indices[y[new_possible_indices] == 0]
                new_indices = np.random.choice(new_possible_indices, counts[1] - counts[0], replace=False)
                indices[-1] = np.concatenate([indices[-1], new_indices])
                indices[-1].sort()

                count_matrix = np.unique(y[indices[-1]], return_counts=True)
                counts = count_matrix[1]

                self.train_history_["train_size"][-1] = np.append(counts, indices[-1].shape[0])

                early_stop = True
                train_last_level = True

            k += 1

        # Those cascade levels beyond the one that reached perfect precision are deleted
        fitted_levels = []
        for k in range(len(self.cascade_levels)):
            level = self.cascade_levels[k]
            fitted_params = [param for param in vars(level.clf) if param[-1] == "_"]
            if len(fitted_params) > 0:
                fitted_levels.append(k)
                continue
        self.cascade_levels = [self.cascade_levels[i] for i in fitted_levels]

    def predict(self, X) -> np.ndarray:
        """
        ### **predict**
        Predicts the target labels given a set of data.

        ###### Inputs
        X
        Contains the explanatory varaibles of the training data.

        ###### Outputs
        y_pred
        Contains the predicted target variables.
        """
        X = np.array(X)
        y_pred = np.empty(X.shape[0])

        indices = []
        indices.append(np.arange(X.shape[0]))

        for level in self.cascade_levels:
            X_test = X[indices[-1], :]

            if callable(getattr(level.clf, "predict_proba", None)) and level.predict_proba:
                y_pred[indices[-1]] = np.where(level.clf.predict_proba(X_test)[:, 0] > level.threshold, 0, 1)
            else:
                y_pred[indices[-1]] = level.clf.predict(X_test)

            indices.append(indices[-1][y_pred[indices[-1]] == 1])

        return y_pred

    def score(self, X, y, save_scores: bool = True) -> List[np.ndarray]:
        """
        ### **score**
        Given a set of data, evaluates the performance of the classifier. Given that the cascade classifier aims to handle heavily umbalanced data, instead of a simple metric, this method returns a list of the total confusion matrices at each cascade level.

        #### Inputs
        ####### X
        Contains the explanatory variables of the data
        ####### y
        Contains the target values of the data

        #### Additional inputs
        ####### save_scores
        Whether the confusion matrices are stored in a class attribute.

        #### Outputs
        ####### cms
        List of the total confusion matrices at each cascade level.
        """
        X = np.array(X)
        y_pred = np.empty(X.shape[0])

        indices = []
        indices.append(np.arange(X.shape[0]))

        cms = []

        for level in self.cascade_levels:
            X_test = X[indices[-1], :]

            if callable(getattr(level.clf, "predict_proba", None)) and level.predict_proba:
                y_pred[indices[-1]] = np.where(level.clf.predict_proba(X_test)[:, 0] > level.threshold, 0, 1)
            else:
                y_pred[indices[-1]] = level.clf.predict(X_test)

            indices.append(indices[-1][y_pred[indices[-1]] == 1])

            cms.append(confusion_matrix(y_true=y, y_pred=y_pred))

        if save_scores:
            self.scores_ = cms
        return cms

    def last_level_gs(self, X, y, parameter_name: str, parameter_list, test_size: float = 0.2) -> Tuple[list, Dict[str, list]]:
        """
        ### **last_level_gs**
        Performs a grid search of a single parameter at the last level of the cascade, and returns the training time and the train and test confusion matrices of the level, for each value of the parameter of interest. Useful to determine the optimal complexity of the last level of the cascade (for example, if the level's classifier is a boosting method, the complexity could be the boosting iterations).

        #### Inputs
        ###### X
        Contains the explanatory variables of the data.
        ###### y
        Contains the target values of the data.
        ###### parameter_name
        The name of the parameter with which the grid search will be performed
        ###### parameter_list
        Iterable object containing the values to test of the parameter of interest.

        #### Additional inputs
        ###### test_size
        The size of the test split. The default value is 0.2.

        #### Outputs
        ###### train_times
        Contains the training times of the level for each value of the parameter of interest
        ###### gs_cms
        Contains the confusion matrices of train and test of the last level for each value of the parameter of interest. These are accesible with the keys: "train_level" and "test_level".
        """
        X = np.array(X)
        y = np.array(y)

        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size)

        if len(self.cascade_levels) > 1:
            sub_cascade_clf = CascadeClassifier(cascade_levels=self.cascade_levels[:-1])
            sub_cascade_clf.fit(X, y)

            y_hat = sub_cascade_clf.predict(X)
            indices = np.arange(X.shape[0])[y_hat == 1]

            y_hat = sub_cascade_clf.predict(X_test)
            indices_test = np.arange(X_test.shape[0])[y_hat == 1]

        else:
            indices = np.arange(X.shape[0])
            indices_test = np.arange(X_test.shape[0])

        X_train = X[indices, :]
        y_train = y[indices]

        X_test = X_test[indices_test, :]
        y_test = y_test[indices_test]

        train_times = []
        gs_cms = {"train_level": [], "test_level": []}

        for parameter in parameter_list:
            level = copy.deepcopy(self.cascade_levels[-1])

            if parameter_name == "threshold":
                thres = parameter
            else:
                thres = level.threshold
                level.clf.set_params(**{parameter_name: parameter})

            t_start = time.time()
            level.clf.fit(X_train, y_train)
            train_times.append(time.time() - t_start)

            if callable(getattr(level.clf, "predict_proba", None)) and level.predict_proba:
                y_hat = np.where(level.clf.predict_proba(X_train)[:, 0] > thres, 0, 1)
                y_hat_test = np.where(level.clf.predict_proba(X_test)[:, 0] > thres, 0, 1)
            else:
                y_hat = level.clf.predict(X_train)
                y_hat_test = level.clf.predict(X_test)

            gs_cms["train_level"].append(confusion_matrix(y_true=y_train, y_pred=y_hat))
            gs_cms["test_level"].append(confusion_matrix(y_true=y_test, y_pred=y_hat_test))

        return train_times, gs_cms

    @staticmethod
    def plot_last_level_gs(
        clf: "CascadeClassifier",
        X,
        y,
        parameter_name: str,
        parameter_list,
        test_size: float = 0.2,
        figsize: float = 7,
        aspect_ratio: float = 1.3,
        suptitle: str = None,
    ):
        """
        ## **plot_last_level_gs**
        Performs a `last_level_gs` and plots the training time and the train and test recall, precision and F1, for each value of the parameter of interest.

        #### Inputs
        ###### clf
        The cascade classifier with which the grid search will be performed.
        ###### X
        The explanatory variables of the data
        ###### y
        The target values of the data
        ###### parameter_name
        The name of the parameter with which the grid search will be performed
        ###### parameter_list
        Iterable object containing the values to test of the parameter of interest.

        #### Additional inputs
        ###### test_size
        The size of the test split. The default value is 0.2.
        ###### figsize
        Height of the figure in inches. Default value is 7.
        ###### aspect_ratio
        Aspect ratio of each of the subplots. Default is 1.3. As the recall, precision and F1 figure has three subplots, the resulting width will be 3*aspect_ratio*figsize.
        ###### suptile
        Title of the figures. As default, there is no title.
        """

        times, cms = clf.last_level_gs(
            X,
            y,
            parameter_name=parameter_name,
            parameter_list=parameter_list,
            test_size=test_size,
        )

        fig, axs = plt.subplots(1, figsize=(aspect_ratio * figsize, figsize))
        if suptitle is not None:
            fig.suptitle(suptitle)

        x_labels = parameter_list

        axs.plot(x_labels, times, ".-")
        axs.set_title("Configuration training time")
        axs.set_ylabel("Training time [s]")
        axs.set_xlabel("Parameter " + parameter_name)

        axs.grid(visible=True)

        plt.show()

        recall_list_train, precision_list_train, f1_list_train, _ = list_normalized_cms(cms["train_level"])
        recall_list_test, precision_list_test, f1_list_test, _ = list_normalized_cms(cms["test_level"])

        fig, axs = plt.subplots(1, 3, figsize=(3 * aspect_ratio * figsize, figsize))
        if suptitle is not None:
            fig.suptitle(suptitle)

        axs[0].plot(x_labels, recall_list_test, ".-", label="Test Recall")
        axs[1].plot(x_labels, precision_list_test, ".-", label="Test Precision")
        axs[2].plot(x_labels, f1_list_test, ".-", label="Test F1")

        axs[0].plot(x_labels, recall_list_train, ".-", label="Train Recall")
        axs[1].plot(x_labels, precision_list_train, ".-", label="Train Precision")
        axs[2].plot(x_labels, f1_list_train, ".-", label="Train F1")

        axs[0].set_ylabel("Recall")
        axs[1].set_ylabel("Precision")
        axs[2].set_ylabel("F1")

        axs[0].set_title("Recall")
        axs[1].set_title("Precision")
        axs[2].set_title("F1")

        for ax in axs:
            ax.set_xlabel("Parameter " + parameter_name)
            ax.legend()
            ax.grid(visible=True)

        plt.show()

    @staticmethod
    def plot_training_time(
        clf: "CascadeClassifier",
        figsize: float = 7,
        aspect_ratio: float = 1.62,
        axes: plt.Axes = None,
        title_time: bool = False,
        label: str = "",
    ) -> plt.Axes:
        """
        ## **plot_training_time**
        Plots the training time of each cascade level of a trained cascade classifier.

        #### Inputs
        ###### clf
        Cascade classifier of which the training times will be plotted. This classifier **must be trained**.

        #### Additional inputs
        ###### figsize
        Height of the figure in inches. Default value is 7.
        ###### aspect_ratio
        Aspect ratio of each of the subplots. Default is 1.62. The resulting width will be aspect_ratio*figsize.
        ###### axes
        Axes to plot the figure. By default this parameter is set to None, and a new set of axes will be created in the method.
        ###### title_time
        Whether the total training time will be included in the title of the figure.
        ###### label
        Label of the plotted line. By default is ''. If it is '' the legend will not be displayed.

        #### Outputs
        ###### axes
        Set of axes used to plot the figures.
        """
        if axes is None:
            fig, axs = plt.subplots(1, figsize=(aspect_ratio * figsize, figsize))
        else:
            axs = axes

        x_labels = np.arange(len(clf.fit_time_))
        title = "Level training time" + (". Total time: %.2f s" % (sum(clf.fit_time_)) if title_time else "")

        axs.plot(
            x_labels,
            clf.fit_time_,
            ".-",
            label=label + " %.2f s" % (sum(clf.fit_time_)),
        )
        axs.set_title(title)
        axs.set_ylabel("Training time [s]")
        axs.set_xlabel("Number of cascade level")

        axs.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
        axs.grid(visible=True)

        if label != "":
            axs.legend()

        if axes is None:
            plt.show()

        return axs

    @staticmethod
    def plot_level_size(clf: "CascadeClassifier", figsize: float = 7, aspect_ratio: float = 1.62) -> plt.Axes:
        """
        ## ***plot_level_size*
        Plots the size of the training dataset's slice that is passed as input to each cascade level.

        #### Inputs
        ###### clf
        Cascade classifier of which the plots will be made from. This classifier **must be trained**.

        #### Additional inputs
        ###### figsize
        Height of the figure in inches. Default value is 7.
        ###### aspect_ratio
        Aspect ratio of each of the subplots. Default is 1.62. The resulting width will be aspect_ratio*figsize.

        #### Outputs
        ###### axes
        Set of axes used to plot the figures.
        """
        history = clf.train_history_["train_size"]

        fig, axs = plt.subplots(1, figsize=(aspect_ratio * figsize, figsize))
        x_labels = np.arange(len(history))

        size_c0 = [m[0] for m in history]
        size_c1 = [m[1] for m in history]
        size_total = [m[2] for m in history]

        axs.plot(x_labels, size_c0, ".-", label="Class 0")
        axs.plot(x_labels, size_c1, ".-", label="Class 1")
        axs.plot(x_labels, size_total, ".-", label="Total")

        axs.set_yscale("log")
        axs.set_xlabel("Number of cascade level")
        axs.set_ylabel("Number of samples")
        axs.set_title("Training dataset size")

        axs.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
        axs.grid(visible=True)

        axs.legend()
        plt.show()

        return axs

    @staticmethod
    def plot_precision_recall_f1(
        clf: "CascadeClassifier",
        X_test=None,
        y_test=None,
        figsize: float = 7,
        aspect_ratio: float = 1.3,
        cms_to_plot: Union[list, str] = None,
        axes: plt.Axes = None,
        label_suffix: str = "",
        suptitle: str = None,
        plot_train_metrics: bool = False,
    ) -> plt.Axes:
        """
        ## **plot_precision_recall_f1**
        Plots the recall, precision and F1 achieved with each cascade level.

        #### Inputs
        ###### clf
        Cascade classifier of which the plots will be made from.
        ###### X_test
        Contains the explanatory variables of the test data.
        ###### y_test
        Contains the target variables of the test data.

        If both X_test and y_test parameters are set to None, the metrics will be loaded from clf.scores_. If this is the case, the classifier must have run its score method before.

        #### Additional inputs
        ###### figsize
        Height of the figure in inches. Default value is 7.
        ###### aspect_ratio
        Aspect ratio of each of the subplots. Default is 1.3. The resulting width will be 3*aspect_ratio*figsize.
        ###### cms_to_plot
        Levels' confusion matrices to display. There are multiple valid values:
            - If set to None, no matrix will be displayed.
            - If set to "all", all confusion matrices from all the cascade levels will be displayed.
            - If a list is passed, this will be interpreted as the indices of the cascade levels of which the confusion matrices will be displayed. For example, if [0, 1, 4] is passed, the confusion matrices corresponding to the first, second and fifth level will be displayed; if [-1] is passed, the confusion matrix corresponding to the last level will be displayed.
        ###### axes
        Axes to plot the figure. By default this parameter is set to None, and a new set of axes will be created in the method.
        ###### label_suffix
        String added at the end of each label of the plot.
        ###### suptitle
        Title of the figure.
        ###### plot_train_metrics
        Whether the train recall, precision and F1 will be plotted. This is set to False as default.

        #### Outputs
        ###### axes
        Set of axes used to plot the figures.
        """
        if X_test is not None and y_test is not None:
            cm_list_test = clf.score(X_test, y_test)
        else:
            if not hasattr(clf, "scores_"):
                raise ValueError("In order to plot the metrics a test dataset is needed or a previous model scoring need to be done.")
            cm_list_test = clf.scores_

        recall_list, precision_list, f1_list, _ = list_normalized_cms(cm_list_test)
        cm_list_test = [normalized_confusion_matrices(matrix) for matrix in cm_list_test]

        if axes is None:
            fig, axs = plt.subplots(1, 3, figsize=(3 * aspect_ratio * figsize, figsize))
            if suptitle is not None:
                fig.suptitle(suptitle)
        else:
            axs = axes

        x_labels = np.arange(len(recall_list))
        axs[0].plot(x_labels, recall_list, ".-", label="Test Recall " + label_suffix)
        axs[1].plot(x_labels, precision_list, ".-", label="Test Precision " + label_suffix)
        axs[2].plot(x_labels, f1_list, ".-", label="Test F1 " + label_suffix)

        axs[0].set_ylabel("Recall")
        axs[1].set_ylabel("Precision")
        axs[2].set_ylabel("F1")

        axs[0].set_title("Recall")
        axs[1].set_title("Precision")
        axs[2].set_title("F1")

        if plot_train_metrics:
            cm_list_train = clf.train_history_["train_cms"]
            recall_list, precision_list, f1_list, _ = list_normalized_cms(cm_list_train)

            x_labels = np.arange(len(recall_list))
            axs[0].plot(x_labels, recall_list, ".--", label="Train Recall " + label_suffix)
            axs[1].plot(x_labels, precision_list, ".--", label="Train Precision " + label_suffix)
            axs[2].plot(x_labels, f1_list, ".--", label="Train F1 " + label_suffix)

        for ax in axs:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
            ax.set_xlabel("Number of cascade level")
            ax.legend()
            ax.grid(visible=True)

        if axes is None:
            plt.show()

        if cms_to_plot is not None:

            if cms_to_plot == "all":
                for cms in cm_list_test:
                    plot_normalized_cms(cms)
            else:
                for cms_level_index in cms_to_plot:
                    plot_normalized_cms(cm_list_test[cms_level_index])

        return axs

    @staticmethod
    def plot_train_metrics(
        clf: "CascadeClassifier",
        figsize: float = 7,
        aspect_ratio: float = 1.3,
        axes: plt.Axes = None,
        label_suffix: str = "",
        suptitle: str = None,
    ) -> plt.Axes:
        """
        # ----------------------- WORK IN PROGRESS -----------------------

        ## **plot_train_metrics**
        Plots the training recall and false positive rate of each cascade level of a cascade classifier. This is used by some authors as a criterion to determine if the model overfits. If the recall decays "much quicker" than the false positive rate, it may mean that there is overfitting.

        #### Inputs
        ###### clf
        Cascade classifier of which the plots will be made from. This classifier **must be trained**.

        #### Additional inputs
        ###### figsize
        Height of the figure in inches. Default value is 7.
        ###### aspect_ratio
        Aspect ratio of each of the subplots. Default is 1.3. The resulting width will be 3*aspect_ratio*figsize.
        ###### axes
        Axes to plot the figure. By default this parameter is set to None, and a new set of axes will be created in the method.
        ###### label_suffix
        String added at the end of each label of the plot.
        ###### suptitle
        Title of the figure.

        #### Outputs
        ###### axes
        Set of axes used to plot the figures.
        """
        check_is_fitted(clf, attributes="train_history_")

        cm_list_train = clf.train_history_["train_cms"]
        recall_list, _, _, fprate_list = list_normalized_cms(cm_list_train)

        if axes is None:
            fig, axs = plt.subplots(1, 3, figsize=(3 * aspect_ratio * figsize, figsize))
            if suptitle is not None:
                fig.suptitle(suptitle)
        else:
            axs = axes

        x_labels = np.arange(len(recall_list))
        axs[0].plot(x_labels, recall_list, ".-")
        axs[1].plot(x_labels, fprate_list, ".-")
        axs[2].plot(x_labels, np.array(recall_list) / np.array(fprate_list), ".-")

        axs[0].set_ylabel("Recall")
        axs[1].set_ylabel("False Positive Rate")
        axs[2].set_ylabel("Ratio")

        axs[0].set_title("Recall")
        axs[1].set_title("False Positive Rate")
        axs[2].set_title("Ratio")

        for ax in axs:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
            ax.set_xlabel("Number of cascade level")
            ax.grid(visible=True)

        if axes is None:
            plt.show()

        return axs

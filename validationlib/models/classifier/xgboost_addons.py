'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import xgboost as xgb


class XGBClf_balanced(xgb.XGBClassifier):
    """
    ## **XGBoost balanced classifier**
    Derived class from xgboost.XGBClassifier to add the `class_weight = 'balanced'` option, in a similar way to several methods in sklearn. This adjusts the weights assigned to positive samples. The weight is set to number_of_negative_class_samples / number_of_positive_class_samples.

    #### Inputs
    ###### class_weight
    If set to 'balanced' the positive samples will have a weight equal to number_of_negative_class_samples / number_of_positive_class_samples in the training data.
    ###### val_size
    Relative size for the validation set.
    ###### kwargs
    xgboost.XGBClassifier keyword arguments.
    """

    def __init__(self, class_weight: str = "balanced", val_size: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.class_weight = class_weight
        self.val_size = val_size

    def fit(self, X, y):
        """
        ### **fit**
        Fits the classifier to a set of data.

        #### Inputs
        ###### X
        Contains the explanatory variables of the training data
        ###### y
        Contains the target values of the training data
        """
        X = np.array(X)
        y = np.array(y)

        if self.val_size == 0:
            X_train = X
            y_train = y
            eval_set = [(X_train, y_train)]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state)
            eval_set = [(X_train, y_train), (X_test, y_test)]

        if self.class_weight == "balanced":
            counts = np.unique(y_train, return_counts=True)[1]
            weight = counts[0] / counts[1]
            self.set_params(scale_pos_weight=weight)

        super().fit(X_train, y_train, eval_set=eval_set, verbose=False)

    @staticmethod
    def plot_validation_results(
        xgb_clf: "XGBClf_balanced",
        figsize: float = 7,
        aspect_ratio: float = 1.62,
        axes: plt.Axes = None,
        add_to_title: str = "",
    ):
        """
        ## **plot_validation_results**
        Plots the validation results with the chosen eval_metric. (See XGBoost documentation for more details)

        #### Inputs
        ###### xgb_clf
        XGBoost balanced classifier of which the validation results will be plotted.

        #### Additional inputs
        ###### figsize
        Height of the figure in inches. Default value is 7.
        ###### aspect_ratio
        Aspect ratio of each of the subplots. Default is 1.62. The resulting width will be aspect_ratio*figsize.
        ###### axes
        Axes to plot the figure. By default this parameter is set to None, and a new set of axes will be created in the method.
        ###### add_to_title
        String to add at the end of the figure title

        #### Outputs
        ###### axes
        Set of axes used to plot the figures.
        """

        results = xgb_clf.evals_result()
        n_iterations = len(results["validation_0"]["logloss"])
        x_axis = range(1, n_iterations + 1)

        if axes is None:
            fig, axs = plt.subplots(1, figsize=(aspect_ratio * figsize, figsize))
        else:
            axs = axes

        axs.plot(x_axis, results["validation_0"]["logloss"], label="Train")
        if xgb_clf.val_size != 0:
            axs.plot(x_axis, results["validation_1"]["logloss"], label="Test")

        axs.legend()
        axs.grid(visible=True)
        axs.set_ylabel("Log Loss")
        axs.set_xlabel("Number of boosting iterations")
        axs.set_title("XGBoost Log Loss. " + add_to_title)

        if axes is None:
            plt.show()

        return axes

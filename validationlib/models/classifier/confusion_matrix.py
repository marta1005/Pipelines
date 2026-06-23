'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''

from typing import List, Tuple

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def normalized_confusion_matrices(cm) -> Tuple[np.ndarray]:
    """
    ## **normalized_confusion_matrices**
    Given a confusion matrix, this function return a tuple of four matrices:
        - The confusion matrix itself.
        - The confusion matrix, where each element is divided by the sum of its row (normalized over true values).
        - The confusion matrix, where each element is divided by the sum of its column (normalized over predicted values).
        - The element-wise harmonic mean of the two previous matrices.

    #### Inputs
    ###### cm
    Confusion matrix

    #### Outputs
    ###### cm
    Confusion matrix
    ###### cm_recall
    Confusion matrix normalized over true values.
    ###### cm_precision
    Confusion matrix normalized over predicted values
    ###### cm_f1
    Element-wise harmonic mean matrix.
    """
    cm_precision = np.divide(cm, np.sum(cm, axis=0)[None, :])
    cm_recall = np.divide(cm, np.sum(cm, axis=1)[:, None])
    cm_f1 = 2 * np.divide(np.multiply(cm_precision, cm_recall), (cm_precision + cm_recall))
    cm = np.array(cm)

    return cm, cm_recall, cm_precision, cm_f1


def list_normalized_cms(cms: List[np.ndarray]) -> Tuple[List[float]]:
    """
    ## **list_normalized_cms**
    Given a list of confusion matrices, this function returns a tuple containing the recall, precision, f1 and false positive rate of those matrices.

    #### Inputs
    ###### cms
    List of confusion matrices from which the outputs will be obtained.

    #### Outputs
    ###### recall_list
    List of recall values of each confusion matrix in the input list.
    ###### precision_list
    List of precision values of each confusion matrix in the input list.
    ###### f1_list
    List of F1 values of each confusion matrix in the input list.
    ###### fprate_list
    List of false positive rate values of each confusion matrix in the input list.
    """
    cm_list = [normalized_confusion_matrices(matrix) for matrix in cms]

    recall_list = [matrix[1] for matrix in cm_list]
    recall_list = [matrix[1, 1] for matrix in recall_list]

    precision_list = [matrix[2] for matrix in cm_list]
    precision_list = [matrix[1, 1] for matrix in precision_list]

    f1_list = [matrix[3] for matrix in cm_list]
    f1_list = [matrix[1, 1] for matrix in f1_list]

    fprate_list = [matrix[1] for matrix in cm_list]
    fprate_list = [matrix[0, 1] for matrix in fprate_list]

    return recall_list, precision_list, f1_list, fprate_list


def plot_normalized_cms(
    cms: list,
    titles: List[str] = [
        "Confusion Matrix",
        "CM Normalized over True Values",
        "CM Normalized over Predicted Values",
        "Harmonic Mean Matrix",
    ],
    fig_size: float = 6,
    aspect_ratio: float = 1.15,
) -> plt.Axes:
    """
    ## **plot_normalized_cms**
    Given a list of normalized confusion matrices, as returned by normalized_confusion_matrices function, it displays each one of the matrices as heatmaps.

    #### Inputs
    ###### cms
    List of normalized confusion matrices. This should be the output of the normalized_confusion_matrices function.
    ###### titles
    List of the names of each matrix. By default is: ["Confusion Matrix", "CM Normalized over True Values", "CM Normalized over Predicted Values", "Harmonic Mean Matrix"]
    ###### figsize
    Height of the figure in inches. Default value is 6.
    ###### aspect_ratio
    Aspect ratio of each of the subplots. Default is 1.15. As this plot will have one subplot for each normalized matrix, the resulting width will be n*aspect_ratio*figsize, being n the number of normalized matrices.

    #### Outputs
    ###### axes
    Set of axes used to plot the figures.
    """
    n = len(cms)
    fig, axs = plt.subplots(1, n, figsize=(n * aspect_ratio * fig_size, fig_size))

    for i in range(n):
        if cms[i].max() <= 1:
            vmax = 1
        else:
            vmax = None

        s = sns.heatmap(
            cms[i],
            annot=True,
            fmt=".4g",
            ax=axs[i],
            vmax=vmax,
            vmin=0,
            cbar_kws={"fraction": 0.15},
        )
        s.set(xlabel="Predicted", ylabel="True", title=titles[i])

    plt.show()

    return axs

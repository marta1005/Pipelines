'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def preprocess_to_cascade(
    X: pd.DataFrame, y: pd.Series, options: object, input_range: pd.DataFrame, output_range: pd.DataFrame
) -> Tuple[pd.DataFrame]:
    """
    Preprocess data for the cascade classifier.

    This function performs the necessary preprocessing steps for the cascade classifier, including encoding categorical variables with cylindrical coordinates and rescaling all variables to the (-1, 1) range.

    :param X: pd.DataFrame
        DataFrame of shape (n_samples, n_features) that contains the explanatory variables.
    :param y: pd.Series
        Series of shape (n_samples,) that contains the target values.
    :param options: object
        Encapsulates the configuration of the surrogate model.
    :param input_range: pd.DataFrame
        Contains the minimum and maximum values for the explanatory variables.
    :param output_range: pd.DataFrame
        Contains the minimum and maximum values for the output variables.

    :return: Tuple[pd.DataFrame]
        A tuple containing the preprocessed explanatory variables (X_pp) and preprocessed output variables (y_pp).

    Notes:
    - The encoding of categorical variables, such as 'Frame' and 'Stringer,' is done with cylindrical coordinates.
    - All variables are rescaled to the (-1, 1) range.
    - Output variables are encoded as 0 or 1 based on whether they fall within the specified output range.

    Example:
    >>> X, y = preprocess_to_cascade(X, y, options, input_range, output_range)
    """

    def frameEncoding(name):
        """
        Maps the frame to a coordinate along the longitudinal axis.

        :param name: str
            The name of the frame to be encoded.
        
        :return: float
            The encoded coordinate along the longitudinal axis.

        Example:
        >>> encoded_frame = frameEncoding("Frame_Name")
        """
        index = options['id_2_order'].index(name)
        return index

    def stringerEncoding(name):
        """
        Maps the stringer to a coordinate between [0, 2*PI].

        :param name: str
            The name of the stringer to be encoded.
        
        :return: float
            The encoded coordinate between [0, 2*PI].

        Example:
        >>> encoded_stringer = stringerEncoding("Stringer_Name")
        """
        index = options['id_1_order'].index(name)
        length = len(options['id_1_order'])
        return 2 * index * np.pi / length

    X_pp = X.copy()
    y_pp = y.copy()

    # dp encoding
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_pp.loc[:, "dp"] = scaler.fit_transform(X_pp[["dp"]])

    # Frame and Stringer encoding
    X_pp.loc[:, "Stringer"] = X_pp.loc[:, "Stringer"].apply(stringerEncoding)
    X_pp.loc[:, "Frame"] = X_pp.loc[:, "Frame"].apply(frameEncoding)

    # Scaling of continuous input variables
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(input_range)
    X_pp.loc[:, input_range.columns] = scaler.transform(X_pp[input_range.columns])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_pp.loc[:, "Stringer"] = scaler.fit_transform(X_pp[["Stringer"]])
    X_pp.loc[:, "Frame"] = scaler.fit_transform(X_pp[["Frame"]])

    # Encoding of output variables
    y_pp[(y_pp >= output_range.loc["max", :]) | (y_pp <= output_range.loc["min", :])] = 0
    y_pp[y_pp != 0] = 1

    return X_pp, y_pp

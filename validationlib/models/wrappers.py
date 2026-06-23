'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''

import re 
import copy
from typing import Union

import os, sys

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

class pyLOMWrapper(BaseEstimator):
    """
    Wrapper for the pyLOM model, enabling its use as a scikit-learn estimator.
    """

    def __init__(self, model, input_scaler, output_scaler, output_names):
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.output_names = output_names

    def fit(self, X, y):
        """
        This function is only for compatibility purposes with scikit-learn estimators. The pyLOM model must be previously fitted.

        :param X: Input data.
        :param y: Target values.

        :return: self
        """
        return self
    
    def predict(self, X:Union[pd.DataFrame, np.ndarray], return_df:bool=False):
        """
        Predicts the output of the model.

        :param X: Input data.
        :param return_df: Whether to return the output as a DataFrame.

        :return: Predictions.
        """

        if isinstance(X, pd.DataFrame):
            indices = X.index
        else:
            indices = None

        from torch import tensor

        X = tensor(np.array(X).astype(np.float32), device=self.model.device)

        X = self.input_scaler.transform(X)
        preds = self.model(X).cpu().detach().numpy()
        preds = self.output_scaler.inverse_transform(preds)

        if return_df:
            preds = pd.DataFrame(preds, columns=self.output_names, index=indices)

        return preds

class S18ModelWrapper(BaseEstimator):
    """
    Wrapper for the S18 ml_deploy model, enabling its use as a scikit-learn estimator.

    Parameters:
    ----------
    - opt: Surrogate options for the ml_deploy model
    - debug: Whether or not to print debug information
    - kwargs_prediction: Dictionary of keyword arguments for the ml_deploy model prediction function
    - input_names: List of input variable names for the ml_deploy model
    - **kwargs: Additional keyword arguments for the ml_deploy model constructor

    """
    def __init__(self, opt, debug=False, kwargs_prediction=None, input_names=None, **kwargs):
        """
        Initialize the S18ModelWrapper.

        :param opt: Surrogate options for the ml_deploy model.
        :param debug: Whether or not to print debug information.
        :param kwargs_prediction: Dictionary of keyword arguments for the ml_deploy model prediction function.
        :param input_names: List of input variable names for the ml_deploy model.
        :param kwargs: Additional keyword arguments for the ml_deploy model constructor.

        :return: None
        """

        from ml_deploy import MLModelProcess

        self.model = MLModelProcess(opt.model_definition_filename, debug=debug, **kwargs)
        self.model.read_model_process()
        self.model.build()

        self.debug = debug
        self.available_outputs = self.model.output_names
        self.output_name = None
        self.kwargs_prediction = kwargs_prediction

        self.opt = opt

        self.input_names = self.model.input_names if input_names is None else input_names

        super().__init__()

    def set_output(self, output_name: str):
        """
        Set the output of the wrapper to a specified output name.

        :param output_name: The name of the output to change to. If set to 'None', all the model outputs will be outputted.

        :return: None
        """
        if output_name is None or output_name in self.available_outputs:
            self.output_name = output_name
        else:
            raise ValueError("Output " + output_name + " is not in the available outputs of the model.")

    def fit(self, X, y):
        """
        This function is only for compatibility purposes with scikit-learn estimators. The ml_deploy model must be previously fitted.

        :param X: Input data.
        :param y: Target values.

        :return: self
        """
        return self

    def predict(self, _X, return_df=False):
        """
        Predict the output of the model given the input data.

        :param _X: Input data. Can be a numpy array or a pandas dataframe.
        :param return_df: Whether or not the returned predictions will be in the shape of a pandas dataframe. By default is set to False.

        :return: y_pred
            The predictions of the model.
        """

        def frame_decoding(index):
            index = int(index)
            name = self.opt.id_2_order[index]
            return name

        def stringer_decoding(index):
            index = int(index)
            name = self.opt.id_1_order[index]
            return name

        def ohe_decoding(X, re_expr, col_name):
            ohe_cols_names = [re.search(re_expr, col).group(0) if re.search(re_expr, col) else col for col in X.columns]
            ohe_cols_idx = [True if re.search(re_expr, col) else False for col in X.columns]
            X.rename(columns={X.columns[i] : ohe_cols_names[i] for i in range(X.shape[1])}, inplace=True)
            col_df = pd.DataFrame(data=X[X.columns[ohe_cols_idx]].idxmax(axis=1), columns=[col_name])
            X = pd.concat([X, col_df], axis=1)
            X.drop(columns=X.columns[ohe_cols_idx + [False]], inplace=True)
            return X

        X = copy.deepcopy(_X)

        if not hasattr(X, "iloc") and X.shape[1] == len(self.input_names):
            X = pd.DataFrame(data=X, columns=self.input_names)

        if 'Stringer' in X.columns:
            if X['Stringer'].dtype != 'O':
                X.loc[:, 'Stringer'] = X.loc[:, "Stringer"].apply(stringer_decoding)
        else:
            X = ohe_decoding(X, r"Str[0-9]{2}p?", "Stringer")

        if 'Frame' in X.columns:
            if X['Frame'].dtype != 'O':
                X.loc[:, 'Frame'] = X.loc[:, "Frame"].apply(frame_decoding)
        else:
            X = ohe_decoding(X, r"Fr[0-9]{2}-Fr[0-9]{2}", "Frame")

        single_row = False
        if X.shape[0] == 1:
            single_row = True
            X = pd.concat([X, X.set_index(X.index + 1)])

        if self.output_name:
            if self.debug:
                y_pred = self.model.predict(X, **self.kwargs_prediction)["PostProcess Results"][self.output_name]
            else:
                y_pred = self.model.predict(X, **self.kwargs_prediction)[self.output_name]
        else:
            if self.debug:
                y_pred = self.model.predict(X, **self.kwargs_prediction)["PostProcess Results"]
            else:
                y_pred = self.model.predict(X, **self.kwargs_prediction)[self.available_outputs]

        if single_row:
            y_pred = y_pred.loc[0:0]

        y_pred = y_pred if return_df else np.array(y_pred)

        del X
        return y_pred
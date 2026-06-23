'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import json

class oneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom one-hot encoder for transforming categorical variables into one-hot encoded format.

    This one-hot encoder allows you to specify which categories you want to encode or
    automatically detect and encode categorical columns. It provides methods for fitting,
    transforming, and inverse transforming one-hot encoded data.

    Parameters:
    ----------
    categories : str or list, default: 'auto'
        The categories to one-hot encode. If 'auto', it will automatically detect
        categorical columns in the input data.

    warnings : bool, default: True
        Whether to show warnings for category mismatches during transformation.

    Methods:
    ----------
    fit_transform(X: pd.DataFrame) -> pd.DataFrame:
        Fit the one-hot encoder and transform the input data.

    transform(X: pd.DataFrame) -> pd.DataFrame:
        Transform the input data using the fitted one-hot encoder.

    inverse_transform(X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        Reverse the one-hot encoding to retrieve the original data.

    load(name: str):
        Load the one-hot encoder from a file.

    save(name: str):
        Save the one-hot encoder to a file.
    """
    def __init__(self, categories='auto', warnings=True):
        """
        Initialize the one-hot encoder.

        :param categories: str or list, default: 'auto'
            The categories to one-hot encode. If 'auto', it will automatically detect categorical columns.
        :param warnings: bool, default: True
            Whether to show warnings for category mismatches.
        """
        self.categories = categories
        self.warnings = warnings

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:     
        """
        Fit the one-hot encoder and transform the input data.

        :param X: pd.DataFrame
            The input data to be one-hot encoded.

        :return: pd.DataFrame
            The one-hot encoded data.
        """   
        # Categories check
        detected = [x for x in X.keys() if x not in X._get_numeric_data().keys()]
        if self.categories=='auto':
            self.categories_ = detected
        else:
            self.categories_ = self.categories
            if self.warnings:
                for x in [x for x in detected if x not in self.categories_]:
                    print(f'Warning - Non-numeric dtype category "{x}" of X missing from passed categories list')
                for x in [x for x in self.categories_ if x not in detected]:
                    print(f'Warning - Numeric dtype category "{x}" of X found on passed categories list')

        # One-hot encode
        self.catvars_ = {}
        for category in X:
            if category not in self.categories_: continue
            self.catvars_[category] = sorted(X[category].unique())
            if self.warnings: print(f'One-hot encoding: "{category}"')
            #Also add categories not present in the current dataframe. CURRENTLY USELESS, but in the future we might want to define the possible variables before fitting!
            encode = pd.get_dummies(X[category].astype(pd.CategoricalDtype(categories=self.catvars_[category]))).add_prefix(f'{category}_')
            X = X.drop(category, axis=1).join(encode)
        
        # Save encoded column names and dimensions
        self.encodedKeys_ = list(X.keys())
        self.encodedDims_ = X.shape[1]
        return X
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted one-hot encoder.

        :param X: pd.DataFrame
            The input data to be one-hot encoded.

        :return: pd.DataFrame
            The one-hot encoded data.
        """

        detected = [x for x in X.keys() if x not in X._get_numeric_data().keys()]
        if self.warnings:
            for x in [x for x in detected if x not in self.categories_]:
                print(f'Warning - Non-numeric dtype category "{x}" of X missing from fitted categories')
            for x in [x for x in self.categories_ if x not in detected]:
                print(f'Warning - Numeric dtype category "{x}" of X found on fitted categories')

        for category in X:
            if category not in self.categories_: continue
            newcatvars = sorted(X[category].unique())
            if self.warnings:
                print(f'One-hot encoding: "{category}"')
                for x in [x for x in newcatvars if x not in self.catvars_[category]]:
                    print(f'Warning - Found variable "{x}" not present in the original dataset')
            #Also add categories not present in the current dataframe
            encoded = pd.get_dummies(X[category].astype(pd.CategoricalDtype(categories=self.catvars_[category]))).add_prefix(f'{category}_')
            X = X.drop(category, axis=1).join(encoded)
        return X
    
    def inverse_transform(self, X: Union[np.ndarray,pd.DataFrame]) -> pd.DataFrame:
        """
        Reverse the one-hot encoding to retrieve the original data.

        :param X: Union[np.ndarray, pd.DataFrame]
            The one-hot encoded data.

        :return: pd.DataFrame
            The original data.
        """
        # Case for X = np.ndarray without labels
        if type(X)!=pd.DataFrame:
            # Dimension check
            if X.shape[1] != self.encodedDims_: raise ValueError(f'Unmatching dimensions {X.shape[1]} (X) / {self.encodedDims_} (fitted)')
            X = pd.DataFrame(
                data = X,
                #index = (list(), list(range(X.shape[1]))),
                columns = self.encodedKeys_
            )

        # Reverse one-hot encode
        for category in self.categories_:
            for catvar in self.catvars_[category]:
                name = f'{category}_{catvar}'
                X.rename(columns={name: catvar}, inplace=True)
            decoded = X[self.catvars_[category]].idxmax(axis=1).rename(category)
            X = X.drop(self.catvars_[category], axis=1).join(decoded)
        return X
    
    def load(self, name: str):
        """
        Load the one-hot encoder from a file.

        :param name: str
            The name of the file to load from.
        """
        with open(f'{name}/{name}--catencoder.json', 'r', encoding='utf-8') as file:
            params = json.load(file)
        self.categories = params['categories']
        self.warnings = params['warnings']
        self.categories_ = params['categories_']
        self.catvars_ = params['catvars_']
        self.encodedKeys_ = params['encodedKeys_']
        self.encodedDims_ = params['encodedDims_']

    def save(self, name: str):
        """
        Save the one-hot encoder to a file.

        :param name: str
            The name of the file to save to.
        """
        params = {
            'categories': self.categories,
            'warnings': self.warnings,
            'categories_': self.categories_,
            'catvars_': self.catvars_,
            'encodedKeys_': self.encodedKeys_,
            'encodedDims_': self.encodedDims_
        }
        with open(f'{name}/{name}--catencoder.json', 'w', encoding='utf-8') as file:
            json.dump(params, file, ensure_ascii=False, indent=4)
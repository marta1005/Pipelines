'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''

from typing import List, Union, Callable, Optional, Tuple

from sklearn.preprocessing import MinMaxScaler
import numpy as np

import json

class minMaxScaler(MinMaxScaler):
    """
    Custom Min-Max Scaler class.

    Extends the scikit-learn MinMaxScaler with additional methods for saving and loading scaler parameters.
    """
    def load(self, name: str):
        """
        Load scaler parameters from a JSON file.

        :param name: str
            The name of the scaler.

        :return:
            None
        """
        with open(f'{name}/{name}--scaler.json', 'r', encoding='utf-8') as file:
            params = json.load(file)
        self.feature_range = params['feature_range']
        self.fit(np.vstack((params['data_max_'],params['data_min_'])))

    def save(self, name: str):
        """
        Save scaler parameters to a JSON file.

        :param name: str
            The name of the scaler.

        :return:
            None
        """
        params = {
            'feature_range': self.feature_range,
            'data_max_': list(self.data_max_),
            'data_min_': list(self.data_min_)
        }
        with open(f'{name}/{name}--scaler.json', 'w', encoding='utf-8') as file:
            json.dump(params, file, ensure_ascii=False, indent=4)
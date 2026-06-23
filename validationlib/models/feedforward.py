'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
from typing import List, Union, Callable, Optional, Tuple
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator
from tensorflow import keras  # TQDM progress bar not working under tf.keras, using normal keras

from . import base as modelutils
from . import scalers as numericNorms
from . import catEncoders

class kerasFeedForward(BaseEstimator):
    """
    A custom feedforward neural network model for regression and classification tasks.
    """
    activationFunctions = {
        'ReLU': keras.layers.ReLU,
        'leakyReLU': keras.layers.LeakyReLU,
        'softmax': keras.layers.Softmax,
        'linear': None,
    }

    def __init__(self,
        nodes: List[int]=None,
        activations: List[str]=None,
        optimizer: str="adam",
        loss: str="mse",
        epochs: int=None,
        batchSize: int=None,
        trainTestSplit: float=0.2,
        callbacks: List[Callable]=[]
    ):
        """
        Parameters:
        ----------
        nodes : list, default: None
            List of integers specifying the number of nodes in each hidden layer.

        activations : list, default: None
            List of activation functions for each hidden layer.

        optimizer : str, default: "adam"
            The optimizer used for model training.

        loss : str, default: "mse"
            The loss function used during training.

        epochs : int, default: None
            The number of training epochs.

        batchSize : int, default: None
            The batch size for training.

        trainTestSplit : float, default: 0.2
            The fraction of data used for testing during training.

        callbacks : list of Callables, default: []
            List of callback functions for custom behavior during training.
        """
        # Build parameters
        self.nodes = nodes
        self.activations = activations
        self.optimizer = optimizer
        self.loss = loss
        # Fit parameters
        self.epochs = epochs
        self.batchSize = batchSize
        self.trainTestSplit = trainTestSplit

        self.numericEncodingX = False
        self.categoricEncodingX = False
        self.numericEncodingY = False
        self.categoricEncodingY = False

        self.callbacks = callbacks
        # Post-fit variables
        self.fitted_ = False

    def defineXnorm(self, 
            minMax: bool=False,
            minMaxData: np.ndarray=None,
            oneHot: bool=False,
            oneHotData: pd.DataFrame=None
        ):
        """
        Define the input data normalization methods.

        :param minMax: bool, default=False
            If True, apply Min-Max scaling to numeric input data.
        
        :param minMaxData: np.ndarray, default=None
            The data used to fit the Min-Max scaler for input data normalization.
        
        :param oneHot: bool, default=False
            If True, apply one-hot encoding to categorical input data.
        
        :param oneHotData: pd.DataFrame, default=None
            The data used to fit the one-hot encoder for input data normalization.
        """
        if minMax:
            self.numericEncodingX = True
            self.numericEncodingX_ = numericNorms.minMaxScaler().fit(minMaxData)
        if oneHot:
            self.categoricEncodingX = True
            self.categoricEncodingX_ = catEncoders.oneHotEncoder().fit(oneHotData)

    def defineYnorm(self,
            minMax: bool=False,
            minMaxData: np.ndarray=None,
            oneHot: bool=False,
            oneHotData: pd.DataFrame=None
        ):
        """
        Define the output data normalization methods.

        :param minMax: bool, default=False
            If True, apply Min-Max scaling to numeric output data.
        
        :param minMaxData: np.ndarray, default=None
            The data used to fit the Min-Max scaler for output data normalization.
        
        :param oneHot: bool, default=False
            If True, apply one-hot encoding to categorical output data.
        
        :param oneHotData: pd.DataFrame, default=None
            The data used to fit the one-hot encoder for output data normalization.

        """
        if minMax:
            self.numericEncodingY = True
            self.numericEncodingY_ = numericNorms.minMaxScaler().fit(minMaxData)
        if oneHot:
            self.categoricEncodingY = True
            self.categoricEncodingY_ = catEncoders.oneHotEncoder().fit(oneHotData)

    def build(self):
        """
        Build the feedforward neural network model with specified architecture and settings.
        """
        self.activation = [self.activationFunctions.get(key) for key in self.activations]

        # Input layer
        self.inputLayer = keras.Input(shape=(self.nodes_[0],), name="Input")  # Specify input shape

        # Hidden layers
        self.hiddenLayer = keras.layers.Dense(self.nodes_[1], name="Hidden-1")(self.inputLayer)
        if self.activation[0] != None:
            self.hiddenLayer = self.activation[0]()(self.hiddenLayer)
        for i, units in enumerate(self.nodes_[2:]):
            self.hiddenLayer = keras.layers.Dense(units, name="Hidden-" + str(i + 2))(self.hiddenLayer)
            if self.activation[i+1] != None:
                self.hiddenLayer = self.activation[i+1]()(self.hiddenLayer)

        # Define full model and compile
        self.model = keras.models.Model(inputs=self.inputLayer, outputs=self.hiddenLayer, name="Autoencoder")
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.summary = self.model.summary
    
    def shapecheck(self,
            X: np.ndarray,
            Y: np.ndarray=None
        ):
        """
        Check if the data dimensions match the model's input and output requirements.

        :param X: np.ndarray
            The input data to be checked.
        
        :param Y: np.ndarray, default=None
            The output data to be checked, if provided.
        """
        if Y is None:
                check = X.shape[1]!=self.nodes_[0]
                if check: raise ValueError(f'Dimensions error - X ({X.shape[1]}) vs input ({self.nodes_[0]}) features')
        else:
            if self.fitted_:
                check = X.shape[1]!=self.nodes_[0] or Y.shape[1]!=self.nodes_[-1]
                if check: raise ValueError(f'Dimensions error - X/Y ({X.shape[1]}/{Y.shape[1]}) vs input/output ({self.nodes_[0]}/{self.nodes_[-1]}) features')
            else:
                self.nodes_ = self.nodes.copy()
                check = X.shape[1]!=self.nodes_[0] or Y.shape[1]!=self.nodes_[-1]
                if check: 
                    print(f'Dimensions error - X/Y ({X.shape[1]}/{Y.shape[1]}) vs input/output ({self.nodes_[0]}/{self.nodes_[-1]}) features')
                    self.nodes_[0], self.nodes_[-1] = X.shape[1], Y.shape[1]
                    print(f'Changing layer configuration to: {self.nodes_}')
        

    def fitcheck(self):
        """
        Check if the model has been fitted before using prediction or transformation methods.
        """
        if self.fitted_==False: raise ValueError('Model has not been fitted yet')
            
    def preprocess(self,
            X: Union[np.ndarray, pd.DataFrame],
            Y: Union[np.ndarray, pd.DataFrame]=None
        ) -> np.ndarray:
        """
        Preprocess input and output data by converting, normalizing, and checking dimensions.

        :param X: Union[np.ndarray, pd.DataFrame]
            The input data to be preprocessed.

        :param Y: Union[np.ndarray, pd.DataFrame], default=None
            The output data to be preprocessed, if provided.

        :return: np.ndarray
            The preprocessed input and output data.
        """
        # Convert dataframes
        if type(X)==pd.DataFrame: 
            if self.categoricEncodingX:
                X = self.categoricEncodingX_.transform(X)
            else: 
                for column in X:
                    if X[column].dtype == 'category': raise ValueError('X contains categories. Use self.defineXnorm')
            X = X.to_numpy()
        
        # Check for types
        X = check_array(X)
        
        # Apply chosen numeric normalization
        if self.numericEncodingX:
            X = self.numericEncodingX_.transform(X)
        

        if Y is not None:
            if type(Y)==pd.DataFrame:
                if self.categoricEncodingY:
                    Y = self.categoricEncodingY_.transform(Y)
                else: 
                    for column in Y:
                        if Y[column].dtype == 'category': raise ValueError('Y contains categories. Use self.defineYnorm')
                Y = Y.to_numpy()

            Y = check_array(Y)

            if self.numericEncodingY:
                Y = self.numericEncodingY_.transform(Y)

        # Check input and output neurons have the correct shape
        self.shapecheck(X, Y)
        return X, Y

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            Y: Union[np.ndarray, pd.DataFrame]
        ):
        """
        Fit the model with training data.

        :param X: Union[np.ndarray, pd.DataFrame]
            The training input data.

        :param Y: Union[np.ndarray, pd.DataFrame]
            The training output data.
        """
        #self.input = X.copy()
        X, Y = self.preprocess(X, Y)
        self.build()
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=self.trainTestSplit, random_state=0)

        #batchSize=None has weird, slow behavior, avoid
        #batchSize>Xtrain.shape[0] induces reshapes down the line, avoid
        if self.batchSize==None or self.batchSize>Xtrain.shape[0]: batchSize = Xtrain.shape[0]
        else: batchSize = self.batchSize
        self.history = self.model.fit(
            Xtrain,
            Ytrain,
            epochs=self.epochs,
            batch_size=batchSize,
            validation_data=(Xtest, Ytest),
            verbose=0,
            callbacks=self.callbacks,
        ).history
        self.fitted_ = True
        return self

    def predict(self, 
        X: Union[np.ndarray, pd.DataFrame], 
        batchSize: int=None
    ) -> Union[pd.DataFrame,np.ndarray]:
        """
        Make predictions using the trained model.

        :param X: Union[np.ndarray, pd.DataFrame]
            The input data for which predictions are to be made.

        :param batchSize: int, default=None
            The batch size for making predictions.

        :return: Union[pd.DataFrame, np.ndarray]
            The predicted output data.
        """
        
        self.fitcheck()
        #self.input = X.copy()
        X, _ = self.preprocess(X)
        if batchSize==None: batchSize = X.shape[0]
        Y = self.model.predict(X, batch_size=batchSize)

        # Apply inverted chosen numeric normalization
        if self.numericEncodingY:
            Y = self.numericEncodingY_.inverse_transform(Y)
        # Apply inverted category encoding
        if self.categoricEncodingY:
            Y = self.categoricEncodingY_.inverse_transform(Y)
        return Y

    def memoryUsage(self, batchSize: int):
        """
        Estimate the memory utilization of the Keras model.

        :param batchSize: int
            The batch size used for estimating memory usage.
        """
        print(f'Estimated memory utilization: {modelutils.memoryUsageKeras(self.model, batch_size = batchSize):.3f} GB')

    def plotLoss(self, figsize=(6, 4)):
        """
        Plot a chart of the model's loss during training.

        :param figsize: tuple, default=(6, 4)
            The size of the loss chart.
        """
        # Plot a loss chart
        plt.figure(figsize=figsize)
        plt.title(label="Model Loss by Epoch", loc="center")

        plt.plot(self.history["loss"], label="Training Data", color="black")
        plt.plot(self.history["val_loss"], label="Test Data", color="red")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def load(self, name: str):
        """
        Load the model's parameters and encoding methods from files.

        :param name: str
            The name of the saved model files to be loaded.
        """
        # Model params
        with open(f'{name}/{name}--params.json', 'r') as file:
            self.set_params(**json.load(file))
        self.nodes_ = self.nodes.copy()

        # Model numeric scaler
        if self.numericEncodingX: 
            self.numericEncodingX_ = numericNorms.minMaxScaler()
            self.numericEncodingX_.load(name+'X')
        if self.numericEncodingY: 
            self.numericEncodingY_ = numericNorms.minMaxScaler()
            self.numericEncodingY_.load(name+'Y')
        # Model category encoder
        if self.categoricEncodingX: 
            self.categoricEncodingX_ = catEncoders.oneHotEncoder()
            self.categoricEncodingX_.load(name+'X')
        if self.categoricEncodingY: 
            self.categoricEncodingY_ = catEncoders.oneHotEncoder()
            self.categoricEncodingY_.load(name+'Y')

        # Model
        self.build()
        self.model.load_weights(f'{name}/{name}--layers.h5')
        self.fitted_ = True

    def save(self, name: str, saveAll: bool=True):
        """
        Save the model's parameters and encoding methods to files.

        :param name: str
            The name to be used for saving the model files.

        :param saveAll: bool, default=True
            If True, save the Keras model weights; otherwise, save only parameters.
        """
        # Create folder if it doesn't exit
        folder = Path().absolute() / name
        Path.mkdir(folder, exist_ok=True)
        
        # Model params
        with open(f'{name}/{name}--params.json', 'w', encoding='utf-8') as file:
            json.dump({**self.get_params(),'nodes': self.nodes_}, file, ensure_ascii=False, indent=4)

        # Model numeric scaler
        if self.numericEncodingX: self.numericEncodingX_.save(name+'X')
        if self.numericEncodingY: self.numericEncodingY_.save(name+'Y')
        # Model category encoder
        if self.categoricEncodingX: self.categoricEncodingX_.save(name+'X')
        if self.categoricEncodingY: self.categoricEncodingY_.save(name+'Y')

        # Model
        if saveAll: self.model.save_weights(f'{name}/{name}--layers.h5')
        print(f'Model successfully saved in {folder}')
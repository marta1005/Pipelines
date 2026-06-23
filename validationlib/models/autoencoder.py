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
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras  # TQDM progress bar not working under tf.keras, using normal keras
import tensorflow as tf

from ..plots import scatterplotMatrix
from . import base as modelutils
from . import scalers as numericNorms
from . import catEncoders

class kerasAutoencoder(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible class for training and using Keras autoencoders.

    Parameters:
    :param nodes: List[int], optional
        Number of neurons in each layer of the autoencoder, including the input and output layers.
    :param activations: List[str], optional
        Activation functions for each layer of the autoencoder.
    :param optimizer: str, optional
        Optimizer for training the autoencoder.
    :param loss: str, optional
        Loss function for training the autoencoder.
    :param sparse: bool, optional
        Whether to use sparse autoencoder.
    :param epochs: int, optional
        Number of training epochs.
    :param batchSize: int, optional
        Batch size for training.
    :param trainTestSplit: float, optional
        Fraction of the data to use for testing.
    :param numericNorm: str, optional
        Numeric data normalization method.
    :param categoryEncoder: str, optional
        Category encoding method.
    :param callbacks: List[Callable], optional
        List of Keras callbacks for training.

    Methods:
    - fit(X): Fit the autoencoder to the input data.
    - transform(X): Transform input data using the trained encoder.
    - predict(X): Reconstruct input data using the trained autoencoder.
    - memoryUsage(batchSize): Print estimated memory utilization.
    - orderNeck(): Order the neurons in the neck layer.
    - plotLoss(figsize): Plot a loss chart.
    - plot2D(marker): Plot 2D visualization of encoded data.
    - plotInputMatrix(X, nShow, method, nbins, s, kdeBins, figsize): Plot input matrix.
    - plotNCmatrix(nShow, method, nbins, s, kdeBins, figsize): Plot neck layer matrix.
    - load(name): Load a saved model.
    - save(name, saveAll): Save the model to a folder.

    Example:
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.preprocessing import StandardScaler
    >>> from keras_autoencoder import kerasAutoencoder
    >>> X, y = make_blobs(n_samples=300, centers=4, random_state=42)
    >>> X = StandardScaler().fit_transform(X)
    >>> ae = kerasAutoencoder(nodes=[2, 8, 4, 2], activations=['ReLU', 'ReLU', 'Neck', 'ReLU'], epochs=100)
    >>> ae.fit(X)
    >>> encoded = ae.transform(X)
    >>> reconstructed = ae.predict(X)
    >>> ae.plotLoss()
    >>> ae.plot2D()
    """

    activationFunctions = {
        'ReLU': keras.layers.ReLU,
        'leakyReLU': keras.layers.LeakyReLU,
        'linear': None,
    }
    numericNorms = {
        'minMax': numericNorms.minMaxScaler()
    }
    categoryEncoders = {
        'oneHot': catEncoders.oneHotEncoder()
    }

    def __init__(self,
        nodes: List[int]=None,
        activations: List[str]=None,
        optimizer: str="adam",
        loss: str="mse",
        sparse: bool=False,
        epochs: int=None,
        batchSize: int=None,
        trainTestSplit: float=0.2,
        numericNorm: str=None,
        categoryEncoder: str=None,
        callbacks: List[Callable]=[]
    ):
        """
        Initialize the kerasAutoencoder.

        :param nodes: List[int], optional
            Number of neurons in each layer of the autoencoder, including the input and output layers.
        :param activations: List[str], optional
            Activation functions for each layer of the autoencoder.
        :param optimizer: str, optional
            Optimizer for training the autoencoder.
        :param loss: str, optional
            Loss function for training the autoencoder.
        :param sparse: bool, optional
            Whether to use a sparse autoencoder.
        :param epochs: int, optional
            Number of training epochs.
        :param batchSize: int, optional
            Batch size for training.
        :param trainTestSplit: float, optional
            Fraction of the data to use for testing.
        :param numericNorm: str, optional
            Numeric data normalization method.
        :param categoryEncoder: str, optional
            Category encoding method.
        :param callbacks: List[Callable], optional
            List of Keras callbacks for training.
        """

        # Build parameters
        self.nodes = nodes
        self.activations = activations
        self.optimizer = optimizer
        self.loss = loss
        self.sparse = sparse
        # Fit parameters
        self.epochs = epochs
        self.batchSize = batchSize
        self.trainTestSplit = trainTestSplit
        self.numericNorm = numericNorm
        self.categoryEncoder = categoryEncoder
        self.callbacks = callbacks
        # Post-fit variables
        self.fitted_ = False

    def build(self):
        """
        Build the architecture of the autoencoder model using the specified parameters.
        """
        self.names = [f"NC {i}" for i in range(1, self.nodes_[0] + 1)]
        self.neckindex = next(i for i, v in enumerate(self.activations) if v=="Neck")
        self.activation = [self.activationFunctions.get(key) for key in self.activations]

        # Input layer
        self.inputLayer = keras.Input(shape=(self.nodes_[0],), name="Input")  # Specify input shape

        # Encoder hidden layers
        self.eLayer = keras.layers.Dense(self.nodes_[1], name="Hidden-1")(self.inputLayer)
        if self.activation[0] != None:
            self.eLayer = self.activation[0]()(self.eLayer)
        for i, units in enumerate(self.nodes_[2 : 1 + self.neckindex]):
            self.eLayer = keras.layers.Dense(units, name="Hidden-" + str(i + 2))(self.eLayer)
            if self.activation[i + 1] != None:
                self.eLayer = self.activation[i + 1]()(self.eLayer)

        # Bottleneck
        if self.sparse:
            self.eLayer = keras.layers.Dense(
                units=self.nodes_[self.neckindex + 1],
                name="Bottleneck",
                activity_regularizer=keras.regularizers.l1(10e-5),)(self.eLayer)
        else:
            self.eLayer = keras.layers.Dense(units=self.nodes_[self.neckindex + 1], name="Neck")(self.eLayer)

        self.dLayer = keras.layers.Dense(self.nodes_[self.neckindex + 2], name="Hidden-" + str(self.neckindex + 2))(self.eLayer)
        if self.activation[self.neckindex + 1] != None:
            self.dLayer = self.activation[self.neckindex + 1]()(self.dLayer)
        for i, units in enumerate(self.nodes_[self.neckindex + 3 :]):
            self.dLayer = keras.layers.Dense(units, name="Hidden-" + str(i + self.neckindex + 3))(self.dLayer)
            if self.activation[i + self.neckindex + 2] != None:
                self.dLayer = self.activation[i + self.neckindex + 2]()(self.dLayer)

        # Define full model and compile
        self.model = keras.models.Model(inputs=self.inputLayer, outputs=self.dLayer, name="SparseAutoencoder")
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.summary = self.model.summary

        # Define encoder and decoder
        self.encoder = keras.models.Model(inputs=self.inputLayer, outputs=self.eLayer)
        self.encoder.compile(optimizer=self.optimizer, loss=self.loss)
    
    def shapecheck(self, X: np.ndarray):
        """
        Check the shape of the input data and verify that it matches the expected input and output layer dimensions.

        :param X: np.ndarray
            Input data.

        :raises: ValueError
            If the shape of the input data does not match the expected input and output layer dimensions.
        """
        if self.fitted_ is False:
            self.nodes_ = self.nodes.copy()
            check = X.shape[1]!=self.nodes_[0] or X.shape[1]!=self.nodes_[-1]
            if check: 
                print(f'Dimensions error - Dataset ({X.shape[1]}) vs input/output ({self.nodes_[0]}/{self.nodes_[-1]}) features')
                self.nodes_[0], self.nodes_[-1] = X.shape[1], X.shape[1]
                print(f'Changing layer configuration to: {self.nodes_}')
        else:
            check = X.shape[1]!=self.nodes_[0] or X.shape[1]!=self.nodes_[-1]
            if check: raise ValueError(f'Dimensions error - Dataset ({X.shape[1]}) vs input/output ({self.nodes_[0]}/{self.nodes_[-1]}) features')

    def fitcheck(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Check whether the model has been fitted and if not, initiate the training process.

        :param X: Union[np.ndarray, pd.DataFrame]
            Input data.
        """
        if self.fitted_ is False:
            print('Model has not been fitted yet, training...')
            self.fit(X)
            
    def preprocess(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Preprocess the input data, including category encoding, conversion to numpy, numeric normalization, and shape verification.

        :param X: Union[np.ndarray, pd.DataFrame]
            Input data.

        :return: np.ndarray
            Preprocessed data.
        """
        # Apply chosen category encoding, convert to numpy
        if self.categoryEncoder is not None:
            if self.fitted_ is False: X = self.categoryEncoder_.fit_transform(X)
            else: X = self.categoryEncoder_.transform(X)
        if type(X)==pd.DataFrame: X = X.to_numpy()
        # Check for types
        X = check_array(X)
        # Check input and output neurons have the correct shape
        self.shapecheck(X)
        # Apply chosen numeric normalization
        if self.numericNorm is not None:
            if self.fitted_ is False: X = self.numericNorm_.fit_transform(X)
            else: X = self.numericNorm_.transform(X)
        return X

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Fit the autoencoder to the input data.

        :param X: Union[np.ndarray, pd.DataFrame]
            Input data.

        :return: kerasAutoencoder
            The trained autoencoder.
        """
        if self.numericNorm is not None: self.numericNorm_ = self.numericNorms.get(self.numericNorm)
        if self.categoryEncoder is not None: self.categoryEncoder_ = self.categoryEncoders.get(self.categoryEncoder)
        self.input = X.copy()
        X = self.preprocess(X)
        self.build()
        Xtrain, Xtest = train_test_split(X, test_size=self.trainTestSplit, random_state=0)

        #batchSize=None has weird, slow behavior, avoid
        #batchSize>Xtrain.shape[0] induces reshapes down the line, avoid
        if self.batchSize==None or self.batchSize>Xtrain.shape[0]: batchSize = Xtrain.shape[0]
        else: batchSize = self.batchSize
        self.history = self.model.fit(
            Xtrain,
            Xtrain,
            epochs=self.epochs,
            batch_size=batchSize,
            validation_data=(Xtest, Xtest),
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
        Reconstruct input data using the trained autoencoder.

        :param X: Union[np.ndarray, pd.DataFrame]
            Input data.
        :param batchSize: int, optional
            Batch size for prediction.

        :return: Union[pd.DataFrame, np.ndarray]
            Reconstructed data.
        """
        
        self.fitcheck(X)
        self.input = X.copy()
        X = self.preprocess(X)
        if batchSize==None: batchSize = X.shape[0]
        X = self.model.predict(X, batch_size=batchSize)

        # Apply inverted chosen numeric normalization
        if self.numericNorm is not None:
            X = self.numericNorm_.inverse_transform(X)
        # Apply inverted category encoding
        if self.categoryEncoder is not None:
            X = self.categoryEncoder_.inverse_transform(X)
        return X
    
    def transform(self, 
        X: Union[np.ndarray, pd.DataFrame], 
        batchSize: int=None
    ) -> np.ndarray:
        """
        Transform input data using the trained encoder.

        :param X: Union[np.ndarray, pd.DataFrame]
            Input data.
        :param batchSize: int, optional
            Batch size for transformation.

        :return: np.ndarray
            Transformed data.
        """
        
        self.fitcheck(X)
        self.input = X.copy()
        X = self.preprocess(X)
        if batchSize==None: batchSize = X.shape[0]
        self.encoded = self.encoder.predict(X, batch_size=batchSize)

        return self.encoded

    def memoryUsage(self, batchSize: int):
        """
        Print estimated memory utilization.

        :param batchSize: int
            Batch size for memory estimation.
        """
        print(f'Estimated memory utilization: {modelutils.memoryUsageKeras(self.model, batch_size = batchSize):.3f} GB')

    def orderNeck(self):
        """
        Order the neurons in the neck layer.
        """
        self.transform(self.input, batchSize=None)
        self.order = np.argsort(-np.mean(np.abs(self.encoded), axis=0))

    def plotLoss(self, figsize=(6, 4)):
        """
        Plot a loss chart.

        :param figsize: tuple, optional
            Figure size.
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

    def plot2D(self, marker: str = "."):
        """
        Plot 2D visualization of encoded data.

        :param marker: str, optional
            Marker style.
        """
        if not(hasattr(self,'order')): self.orderNeck()
        idx = self.order[0:2]
        plt.xlabel(self.names[0])
        plt.ylabel(self.names[1])
        plt.scatter(self.encoded[:, idx[0]], self.encoded[:, idx[1]], marker=marker)

    def plotInputMatrix(self,
        X: Union[pd.DataFrame,np.ndarray],
        nShow: int, 
        method: str, 
        nbins: int = 10, 
        s: float = 1, 
        kdeBins: int = 4, 
        figsize: float = 8,
    ):
        """
        Plot input matrix.

        :param X: Union[pd.DataFrame, np.ndarray]
            Input data.
        :param nShow: int
            Number of features to show.
        :param method: str
            Plotting method.
        :param nbins: int, optional
            Number of bins for histogram.
        :param s: float, optional
            Scatterplot marker size.
        :param kdeBins: int, optional
            Number of bins for kernel density estimation.
        :param figsize: float, optional
            Figure size.
        """
        
        fig = scatterplotMatrix(
            np.array(X)[:, 0:nShow],
            [i for i in X.columns][0:nShow], 
            method, 
            nbins=nbins, 
            s=s, 
            kdeBins=kdeBins, 
            figsize=figsize)
        plt.show()

    def plotNCmatrix(self,
        nShow: int,
        method: str,
        nbins: int = 10,
        s: float = 1,
        kdeBins: int = 4,
        figsize: float = 8
    ):
        """
        Plot neck layer matrix.

        :param nShow: int
            Number of features to show.
        :param method: str
            Plotting method.
        :param nbins: int, optional
            Number of bins for histogram.
        :param s: float, optional
            Scatterplot marker size.
        :param kdeBins: int, optional
            Number of bins for kernel density estimation.
        :param figsize: float, optional
            Figure size.
        """
        if not(hasattr(self,'order')): self.orderNeck()
        if nShow > self.nodes_[self.neckindex+1]: raise ValueError(f'nshow ({nShow}) > neurons at the neck ({self.nodes_[self.neckindex+1]})')
        idx = self.order[0:nShow]
        fig = scatterplotMatrix(
            self.encoded[:, idx],
            self.names[0:nShow],
            method,
            nbins=nbins,
            s=s,
            kdeBins=kdeBins,
            figsize=figsize)
        plt.show()
    
    def load(self, name: str):
        """
        Load a saved model.

        :param name: str
            Name of the saved model.
        """
        # Model params
        with open(f'{name}/{name}--params.json', 'r') as file:
            self.set_params(**json.load(file))
        self.nodes_ = self.nodes.copy()

        # Model numeric scaler
        if self.numericNorm is not None: 
            self.numericNorm_ = self.numericNorms.get(self.numericNorm)
            self.numericNorm_.load(name)
        # Model category encoder
        if self.categoryEncoder is not None: 
            self.categoryEncoder_ = self.categoryEncoders.get(self.categoryEncoder)
            self.categoryEncoder_.load(name)
        
        # Model
        self.build()
        self.model.load_weights(f'{name}/{name}--layers.h5')
        self.fitted_ = True

    def save(self, name: str, saveAll: bool=True):
        """
        Save the model to a folder.

        :param name: str
            Name of the folder.
        :param saveAll: bool, optional
            Whether to save all model weights.
        """
        # Create folder if it doesn't exit
        folder = Path().absolute() / name
        Path.mkdir(folder, exist_ok=True)
        
        # Model params
        with open(f'{name}/{name}--params.json', 'w', encoding='utf-8') as file:
            json.dump({**self.get_params(),'nodes': self.nodes_}, file, ensure_ascii=False, indent=4)

        # Model numeric scaler
        if self.numericNorm is not None: self.numericNorm_.save(name)
        # Model category encoder
        if self.categoryEncoder is not None: self.categoryEncoder_.save(name)

        # Model
        if saveAll: self.model.save_weights(f'{name}/{name}--layers.h5')
        print(f'Model successfully saved in {folder}')
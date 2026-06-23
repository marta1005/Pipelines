'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''

from typing import List, Union, Callable, Optional, Tuple

from ..plots import scatterplotMatrix
from ..plots.common import multiplePlots
from . import scalers as numericNorms
from . import catEncoders

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA as skl_SparsePCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

import json
from pathlib import Path

class pca(BaseEstimator, TransformerMixin):
    """
    A custom Principal Component Analysis (PCA) transformer for dimensionality reduction.
    """
    numericNorms = {
        'minMax': numericNorms.minMaxScaler()
    }
    categoryEncoders = {
        'oneHot': catEncoders.oneHotEncoder()
    }

    def __init__(self,
        nComponents: int=None,
        numericNorm: str=None,
        categoryEncoder: str=None
        ):
        """
        Initialize the PCA transformer.

        :param nComponents: int, default=None
            The number of principal components to retain.

        :param numericNorm: str, default=None
            The method for numeric normalization.

        :param categoryEncoder: str, default=None
            The method for encoding categorical variables.
        """
        # Build parameters
        self.nComp = nComponents
        # Fit parameters
        self.numericNorm = numericNorm
        self.categoryEncoder = categoryEncoder
        # Post-fit variables
        self.fitted_ = False

    def build(self):
        """
        Build the PCA model with specified settings.
        """
        self.names = [f"PC {i}" for i in range(1, self.nComp + 1)]
        self.model = PCA(n_components=self.nComp)
            
    def preprocess(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Preprocess input data by applying chosen category encoding and numeric normalization.

        :param X: Union[np.ndarray, pd.DataFrame]
            The input data to be preprocessed.

        :return: np.ndarray
            The preprocessed input data.
        """
        # Apply chosen category encoding, convert to numpy
        if self.categoryEncoder is not None:
            if self.fitted_ is False: X = self.categoryEncoder_.fit_transform(X)
            else: X = self.categoryEncoder_.transform(X)
        if type(X)==pd.DataFrame: X = X.to_numpy()
        # Check for types
        X = check_array(X)
        # Apply chosen numeric normalization
        if self.numericNorm is not None:
            if self.fitted_ is False: X = self.numericNorm_.fit_transform(X)
            else: X = self.numericNorm_.transform(X)
        return X

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Fit and transform the input data using PCA.

        :param X: Union[np.ndarray, pd.DataFrame]
            The input data.

        :return: pd.DataFrame
            The transformed data.
        """
        if self.numericNorm is not None: self.numericNorm_ = self.numericNorms.get(self.numericNorm)
        if self.categoryEncoder is not None: self.categoryEncoder_ = self.categoryEncoders.get(self.categoryEncoder)
        self.input = X.copy()
        X = self.preprocess(X)
        self.build()

        X = self.model.fit_transform(self.input)
        self.encoded = pd.DataFrame(data=X, columns=self.names)

        self.explainedVariance_ = self.model.explained_variance_ratio_
        self.cumulativeVariance_ = self.explainedVariance_.cumsum()
        self.fitted_ = True
        return self.encoded

    def transform(self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Transform the input data using the pre-fitted PCA model.

        :param X: Union[np.ndarray, pd.DataFrame]
            The input data.

        :return: pd.DataFrame
            The transformed data.
        """
        
        self.input = X.copy()
        X = self.preprocess(X)
        X = self.model.transform(X)
        self.encoded = pd.DataFrame(data=X, columns=self.names)
        return self.encoded

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the PCA-reduced data to the original space.

        :param X: np.ndarray
            The input data in PCA space.

        :return: np.ndarray
            The inverse-transformed data.
        """
        X = self.model.inverse_transform(X)
        # Apply inverted chosen numeric normalization
        if self.numericNorm is not None:
            X = self.numericNorm_.inverse_transform(X)
        # Apply inverted category encoding
        if self.categoryEncoder is not None:
            X = self.categoryEncoder_.inverse_transform(X)
        return X

    def variancePlot(self):
        """
        Plot a chart of the explained variance for each principal component.
        """
        x, y = [i for i in range(1, self.nComp + 1)], self.explainedVariance_ * 100
        plt.plot(x, y, "-o")
        for i, txt in enumerate(y):
            if round(txt,2)==0: continue
            plt.annotate(f'{txt:.2f}', (x[i], y[i]), fontsize=7, weight='bold')
        plt.xticks(list(range(1,self.nComp+1)))
        plt.xlabel("Component")
        plt.ylabel("Percentage explained variance")
        plt.title("Explained Variance")
        plt.grid(visible=True)
        plt.show()

    def cumulativePlot(self):
        """
        Plot a chart of the cumulative explained variance.
        """
        x, y = [i for i in range(1, self.nComp + 1)], self.cumulativeVariance_ * 100
        plt.plot(x, y, "-o")
        for i, txt in enumerate(y): 
            if round(txt,2)==100: continue
            plt.annotate(f'{txt:.2f}', (x[i], y[i]), fontsize=7, weight='bold')
        plt.xticks(list(range(1,self.nComp+1)))
        plt.xlabel("Component")
        plt.ylabel("Percentage explained variance")
        plt.title("Cumulative Variance")
        plt.grid(visible=True)
        plt.show()

    def plot2D(self, marker: str = "."):
        """
        Plot a 2D scatter plot of the first two principal components.

        :param marker: str, default="."
            Marker style for the scatter plot.
        """
        if not(hasattr(self,'encoded')): self.transform(self.input)
        plt.xlabel(self.names[0])
        plt.ylabel(self.names[1])
        plt.scatter(self.encoded.iloc[:, 0], self.encoded.iloc[:, 1], marker=marker)

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
        Plot a matrix of scatter plots for input variables.

        :param X: Union[pd.DataFrame, np.ndarray]
            The input data.

        :param nShow: int
            The number of input variables to show in the matrix.

        :param method: str
            The method for plotting.

        :param nbins: int, default=10
            The number of bins for histograms.

        :param s: float, default=1
            Marker size for scatter plots.

        :param kdeBins: int, default=4
            Number of bins for kernel density estimation.

        :param figsize: float, default=8
            The size of the figure.
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

    def plotPCmatrix(self,
        nShow: int,
        method: str,
        nbins: int = 10,
        s: float = 1,
        kdeBins: int = 4,
        figsize: float = 8
    ):
        """
        Plot a matrix of scatter plots for principal components.

        :param nShow: int
            The number of principal components to show in the matrix.

        :param method: str
            The method for plotting.

        :param nbins: int, default=10
            The number of bins for histograms.

        :param s: float, default=1
            Marker size for scatter plots.

        :param kdeBins: int, default=4
            Number of bins for kernel density estimation.

        :param figsize: float, default=8
            The size of the figure.
        """
        if not(hasattr(self,'encoded')): self.transform(self.input)
        if nShow > self.nComp: raise ValueError(f'nshow ({nShow}) > calculated principal components ({self.nComp})')
        fig = scatterplotMatrix(
            np.array(self.encoded)[:, 0:nShow],
            self.names[0:nShow],
            method,
            nbins=nbins,
            s=s,
            kdeBins=kdeBins,
            figsize=figsize)
        plt.show()
    
    # ON HOLD in case jsonpickle becomes an alternative
    # def load(self, name: str):
    #     # Model params
    #     with open(f'{name}/{name}--params.json', 'r') as file:
    #         self.set_params(**json.load(file))
    #     self.nodes_ = self.nodes.copy()

    #     # Model numeric scaler
    #     if self.numericNorm is not None: 
    #         self.numericNorm_ = self.numericNorms.get(self.numericNorm)
    #         self.numericNorm_.load(name)
    #     # Model category encoder
    #     if self.categoryEncoder is not None: 
    #         self.categoryEncoder_ = self.categoryEncoders.get(self.categoryEncoder)
    #         self.categoryEncoder_.load(name)
        
    #     # Model
    #     self.build()
    #     self.model.load_weights(f'{name}/{name}--layers.h5')
    #     self.fitted_ = True

    # def save(self, name: str, saveAll: bool=False):
    #     # Create folder if it doesn't exit
    #     folder = Path().absolute() / name
    #     Path.mkdir(folder, exist_ok=True)
        
    #     # Model params
    #     with open(f'{name}/{name}--params.json', 'w', encoding='utf-8') as file:
    #         json.dump({**self.get_params(),'nodes': self.nodes_}, file, ensure_ascii=False, indent=4)

    #     # Model numeric scaler
    #     if self.numericNorm is not None: self.numericNorm_.save(name)
    #     # Model category encoder
    #     if self.categoryEncoder is not None: self.categoryEncoder_.save(name)

    #     # Model
    #     if saveAll: self.model.save_weights(f'{name}/{name}--layers.h5')
    #     print(f'Model successfully saved in {folder}')

class GroupedPCA(BaseEstimator, TransformerMixin):
    """
    ## **Grouped PCA**
    Performs PCA by groups of variables

    #### Inputs
    ###### n_components
    Number of components to maintain from each PCA. This can be either a single integer, a list of integers or 'all'.
    - If set to a single number, every PCA will only maintain n_components components
    - If set to a list (with the same number of elements as the number of groups), each group will maintain the corresponding number of components from the list
    - If set to 'all', each group will maintain all of its variables
    ###### numeric_norm
    The kind of norm applied to numeric variables. For now, it only supports 'minMax' to scale each feature to the range [0, 1].
    """
    numeric_norms = {
        'minMax' : numericNorms.minMaxScaler()
    }

    def __init__(self, n_components: Union[int, List[int], str], numeric_norm: str, sparse: bool = False, transformer_kwargs: dict = {}):
        """
        Initialize the Grouped PCA transformer.

        :param n_components: Union[int, List[int], str]
            Number of components to maintain from each PCA.

        :param numeric_norm: str
            The kind of norm applied to numeric variables.

        :param sparse: bool, default=False
            Whether to use Sparse PCA.

        :param transformer_kwargs: dict, default={}
            Additional keyword arguments for the transformer.
        """
        self.n_components = n_components
        self.numeric_norm = self.numeric_norms[numeric_norm]
        self.sparse = sparse

        self.__transformer_kwargs = transformer_kwargs

        if sparse:
            self.__transformer = SparsePCA
            self.__transformer_kwargs['normalize_components'] = True
        else:
            self.__transformer = PCA

    def preprocess(self, X):
        """
        ## **preprocess**

        Preprocesses the data with the defined numeric norm.

        :param X:
            Input data.

        :return:
            Preprocessed data.
        """
        if self.numeric_norm is not None:
            if hasattr(self, "norm_fitted_"):
                X = self.numeric_norm.transform(X)
            else:
                self.norm_fitted_ = True
                X = self.numeric_norm.fit_transform(X)
        return X

    def fit(self, X, variable_groups: Union[List[int], List[str]]):
        """
        ## **fit**
        Fits the PCAs.

        :param X:
            Input data. Can be a numpy array or a pandas dataframe.

        :param variable_groups: Union[List[int], List[str]]
            Groups of variables to apply a PCA to.
        """
        if isinstance(X, pd.DataFrame):
            self.variable_groups_ = [[X.columns.get_loc(col) for col in group] for group in variable_groups]            
            flattened_groups = [col for group in variable_groups for col in group]
            unaltered_columns = [col for col in X.columns if col not in flattened_groups]
            self.unaltered_columns_ = [X.columns.get_loc(col) for col in unaltered_columns]
            self.pca_columns_ = [X.columns.get_loc(col) for col in flattened_groups]
        else:
            self.variable_groups_ = variable_groups
            self.pca_columns_ = [col for group in variable_groups for col in group]
            self.unaltered_columns_ = [col for col in range(X.shape[1]) if col not in self.pca_columns_]

        X = np.array(X)

        self.no_num_cols_ = []
        self.num_cols_ =  []

        for col in range(X.shape[1]):
            try:
                X[:, col].astype(np.float64)
                self.num_cols_.append(col)
            except:
                if col in self.pca_columns_:
                    raise ValueError("Grouped variables for PCA must be numeric")
                else:
                    self.no_num_cols_.append(col)

        self.pcas_ = []
        self.n_features_ = []
        X[:, self.num_cols_] = self.preprocess(X[:, self.num_cols_]) 

        for i, group in enumerate(self.variable_groups_):
            self.n_features_.append(len(group))
            n_components = self.__get_n_components(i)
            #self.pcas_.append(PCA(n_components=n_components))
            self.pcas_.append(self.__transformer(n_components=n_components, **self.__transformer_kwargs))
            self.pcas_[i].fit(X[:, group])
            
        self.fitted = True

    def transform(self, X):
        """
        Performs the transformation from input space to each PCA space (one for each group of variables), 
        and returns n_components components from each group. 
        The input variables not specified in variable_groups are ignored and returned as they were (if they are not numeric) 
        or scaled (if the are numeric).

        :param X:
            Input data to transform.

        :return:
            Transformed data.
        """
        X = np.array(X)
        n_components = self.__get_n_components()
        transformed_X = np.empty([X.shape[0], n_components])
        X[:, self.num_cols_] = self.preprocess(X[:, self.num_cols_])
        for i, group in enumerate(self.variable_groups_):
            n_components = self.__get_n_components(i)
            start_idx = sum(self.__get_n_components(j) for j in range(i))
            end_idx = sum(self.__get_n_components(j) for j in range(i+1))
            
            transformed_X[:, start_idx:end_idx] = self.pcas_[i].transform(X[:, group])

        output_X = np.concatenate([transformed_X, X[:, self.unaltered_columns_]], axis=1)
        return output_X

    def inverse_transform(self, X):
        """
        Performs the inverse transform of each PCA, and the inverse transform of the scaling, returning the reconstructed variables in input space.

        :param X:
            Input data in PCA space.

        :return:
            Reconstructed data in input space.
        """
        inverse_X = np.empty([X.shape[0], sum(self.n_features_)+len(self.unaltered_columns_)], dtype=np.object)
        for i, group in enumerate(self.variable_groups_):
            n_components = self.__get_n_components(i)
            start_idx = sum(self.__get_n_components(j) for j in range(i))
            end_idx = sum(self.__get_n_components(j) for j in range(i+1))

            start_idx_inv = sum(self.n_features_[:i])
            end_idx_inv = sum(self.n_features_[:i+1])

            inverse_X[:, group] = self.pcas_[i].inverse_transform(X[:, start_idx:end_idx])

        if len(self.unaltered_columns_):
            inverse_X[:, self.unaltered_columns_] = X[:, -len(self.unaltered_columns_):]

        if self.numeric_norm is not None:
            inverse_X[:, self.num_cols_] = self.numeric_norm.inverse_transform(inverse_X[:, self.num_cols_])
        return inverse_X

    def get_explained_variance(self, X) -> float:
        """
        Get the total explained variance in PCA space.

        :param X:
            Input data.

        :return:
            Total explained variance.
        """
        X = np.array(X)
        X_reconstructed = self.inverse_transform(self.transform(X))
        return X_reconstructed[:, self.num_cols_].var() / X[:, self.num_cols_].var()

    def get_reconstruction_error(self, X, by_feature:bool=False, err:str='rmse'):
        """
        Return the reconstruction error in input variable units.
        Type of error returned. This can be:
        - 'max': Maximum reconstruction error present in the input data
        - 'rmse': Root Mean Squared Error
        - 'all': Reconstruction error of each sample and variable
        ###### by_feature
        If err=='rmse' or err=='max', this parameter decides whether the returned reconstruction error is given by variable or not.

        :param X:
            Input data.

        :param by_feature: bool, default=False
            Whether to return reconstruction error by feature.

        :param err: str, default='rmse'
            Type of error returned.
        """
        X = np.array(X)
        axis = 0 if by_feature else None

        X_rec = self.inverse_transform(self.transform(X))

        if err == 'max':
            return np.max(np.abs(X[:, self.pca_columns_] - X_rec[:, self.pca_columns_]), axis=axis)
        elif err == 'rmse': 
            return np.sqrt(np.mean(np.square(X[:, self.pca_columns_].astype(np.float64) - X_rec[:, self.pca_columns_].astype(np.float64)), axis=axis))
        elif err == 'all':
            return X[:, self.pca_columns_] - X_rec[:, self.pca_columns_]
        
    def plot_reconstruction_error(self, X, feature_names : List[str] = None, nWorkers:int=1, figHsize:float=18, figAspectRatio:float=4.5):
        """
        Scatterplots for each input variable and its reconstruction error.

        :param X:
            Input data.

        :param feature_names: List[str], default=None
            List of variable names.

        :param nWorkers: int, default=1
            Number of parallel workers for plotting.

        :param figHsize: float, default=18
            Figure size along the horizontal axis.

        :param figAspectRatio: float, default=4.5
            Figure aspect ratio.
        """

        X = np.array(X)

        rec_error = self.get_reconstruction_error(X, err='all')

        nPlots = X.shape[1]

        def func(i, axes:plt.Axes):
            axes.scatter(X[:, self.num_cols_][:, self.pca_columns_][:, i], rec_error[:, i], s=1)
            if feature_names is not None:
                axes.set_xlabel(feature_names[i])

            return axes, None

        fig = multiplePlots(
            nPlots,
            func,
            nWorkers,
            figHsize=figHsize,
            figAspectRatio=figAspectRatio,
            xlabel=None,
            ylabel="Reconstruction error",
            oneColorbar=False,
            tight_layout=False,
        )

        return fig

    def __get_n_components(self, i:int=None):
        """
        Internal auxiliar function to get the number of components of each group of variables in different cases.

        :param i: int, default=None
            The index of the group of variables.

        :return:
            Number of components.
        """
        if self.n_components == "all":
            if i is None: return sum(self.n_features_)
            else: return self.n_features_[i]

        elif hasattr(self.n_components, "__iter__"):
            if i is None: return sum(self.n_components)
            else: return self.n_components[i]

        else:
            if i is None: return self.n_components*len(self.variable_groups_)
            else: return self.n_components
            

class SparsePCA(skl_SparsePCA):
    def inverse_transform(self, X):
        X = np.array(X)

        return (X @ self.components_) + self.mean_
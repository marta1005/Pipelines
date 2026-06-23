'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
"""
Module to manage metrics.
The metrics are defined pointwise.  If you neeed other metrics needs to be 
further developed

For supporting complex metrics is used sympy.
"""
from abc import ABC
from typing import Union, Callable

import sympy
from sklearn.utils import check_consistent_length

predefinedDistanceMetrics = [
    "ratio", 
    "percentage", 
    "residual", 
    "abs_residual"
]

class DistanceMetrics(ABC):
    """
    Abstract base class for defining distance metrics used for evaluating model predictions.

    The DistanceMetrics class allows you to define custom distance metrics, 
    or use predefined ones like ratio, percentage, residual, and absolute residual.

    :param name: Displayed metric name, used for identification.
    
    Attributes:
    :attribute name: The name of the distance metric.
    :attribute y_true: Symbolic variable representing true values.
    :attribute y_pred: Symbolic variable representing predicted values.

    Methods:
    :method __call__: Calculate the distance metric given true and predicted values.
    :method define_metric: Define a specific distance metric expression based 
    on a predefined string or custom expression.

    Properties:
    :property expr: The mathematical expression representing the distance metric.
    :property neutral: The neutral value of the distance metric.

    Example Usage:
    >>> metric = DistanceMetrics("CustomMetric")
    >>> metric.define_metric("y_true - y_pred")
    >>> distance = metric(y_true, y_pred)
    """
    def __init__(self, name: str):
        """
        Initialize a DistanceMetrics instance with a specified metric name.

        :param name: The name of the distance metric.
        """
        self.name = name

        self.y_true = sympy.Symbol('y_true')
        self.y_pred = sympy.Symbol('y_pred')
        # Default metrix is residual
        self.expr = self.y_true - self.y_pred
        self.metric = sympy.lambdify([self.y_true, self.y_pred], self.expr)
        self.neutral: float = self.metric(1,1)
        # Sign if to know if conservative is higher than the neutral (+1) or
        # lower (-1)
        self.sign: float = 1.0

    def __call__(self, y_true, y_pred):
        """
        Calculate the distance metric given true and predicted values.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :return: Calculated distance metric.
        """
        check_consistent_length([y_true, y_pred])
        return self.metric(y_true, y_pred)

    def define_metric(self, expression: Union[str,Callable] = None):
        """
        Define a specific distance metric expression based on a predefined string or
        a custom expression.

        :param expression: A predefined string or a custom expression for the distance metric.
        :return: The DistanceMetrics instance with the defined metric expression.
        """
        if isinstance(expression, str):
            if expression in predefinedDistanceMetrics:
                if expression == "ratio":
                    self.expr = self.y_pred / self.y_true
                elif expression == "percentage":
                    self.expr = self.y_true / self.y_pred - 1
                elif expression == "residual":
                    self.expr = self.y_true - self.y_pred
                elif expression == "abs_residual":
                    self.expr = abs(self.y_true - self.y_pred)
            else: # Custom metric, like 'y_true - y_pred'
                self.expr = sympy.parsing.sympy_parser.parse_expr(expression, evaluate=False)
        elif isinstance(expression, Callable):
            self.expr = expression(y_true=self.y_true, y_pred=self.y_pred)

        self.metric = sympy.lambdify([self.y_true, self.y_pred], self.expr)

        self.neutral = self.metric(1,1)

        return self

# register_distribution_metrics = {"mean", "std", "percentile", "max", "min", "sum",
#                                  "var", "std", "skew", "kurtosis", "median", "iqr"}
# register_global_metrics = get_scorer_names()

# class GlobalMetrics(ABC):
#     """
#     Base class to define the interfaces for all metrics.
#     There is the concept of global metric similar to sklearn (MAE, MAPE, ...)
#     And the concept of pointwise distance between true and prediction
#     """
#     def __init__(global_metrics: List[str], dist_metrics: List[str] = [], **kwargs):
#         self.name = "BaseGlobal"
#         self.distance_metric = None
#         not_allowed_metrics = set(global_metrics) - set(register_global_metrics)
#         if len(not_allowed_metrics) > 0:
#             raise ValueError(f"Not allowed global metrics: {not_allowed_metrics}")

#         self.global_metrics = global_metrics
#         not_allowed_metrics = set(dist_metrics) - register_distribution_metrics
#         if len(not_allowed_metrics) > 0:
#             raise ValueError(f"Not allowed distribution metrics: {not_allowed_metrics}")
#         self.dist_metrics = dist_metrics
#         self.__dict__.update(kwargs)

#     def _score_distribution(y_distribution: pd.DataFrame) -> float:
#         if not isinstance(y_distribution, pd.DataFrame):
#             return 0
#         else:
#             results = []
#             for metric in self.dist_metrics:
#                 if metric == "mean":
#                     results += [y_distribution.mean(axis=0)]
#                 elif metric == "sum":
#                     results += [y_distribution.sum(axis=0)]
#                 elif metric == "max":
#                     results += [y_distribution.max(axis=0)]
#                 elif metric == "min":
#                     results += [y_distribution.min(axis=0)]
#                 elif metric == "std":
#                     results += [y_distribution.std(axis=0)]
#                 elif metric == "var":
#                     results += [y_distribution.var(axis=0)]
#                 elif metric == "skew":
#                     results += [y_distribution.skew(axis=0)]
#                 elif metric == "median":
#                     results += [y_distribution.median(axis=0)]
#                 elif metric == "kurtosis" :
#                     results += [y_distribution.kurtosis(axis=0)]
#                 elif metric == "percentile":
#                     if "quantiles" in self.__dict__.keys():
#                         quantiles = self.quantiles
#                     else:
#                         quantiles = [0.25, 0.5, 0.75]
#                     results += [y_distribution.quantile(q=quantiles, method="table")]
#                 elif metric == "iqr":
#                     quantiles = [0.25, 0.75]
#                     y_distribution.quantile(q=quantiles, method="table")
#                     results += [y_distribution[0.75] - y_distribution[0.25]]
#                 else:
#                     print(f"Unknown global metric: {metric}")
#             return results

#     def _score_global(y_true: pd.DataFrame, y_pred: pd.DataFrame, sample_weights = None) -> float:
#             if not isinstance(y_true, pd.DataFrame):
#                 return 0
#             else:
#                 results = []
#                 for metric in self.global_metrics:
#                     if metric in registered_global_metrics:
#                         results += [eval(metric)(y_true, y_pred, sample_weights)]
#                 return results

#     def __call__(y_true: pd.DataFrame, y_pred: pd.DataFrame = None,
#                 sample_weights = None) -> float:
#         metrics = []
#         if y_pred is None:
#             if self.distance_metric is None:
#                 all_distances = pd.DataFrame()
#             else:
#                 all_distances = y_true
#         else:
#             metrics += [self._score_global(y_true, y_pred, sample_weights)]
#             all_distances = self.distance_metric(y_true, y_pred)
#         metrics += [self._score_distribution(all_distances)]
#         return metrics

#TODO Generalize the concept to include confidence interval through
#     non-parametric, bootstraping or fitted distribution

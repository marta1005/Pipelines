from typing import List, Union
import sys
import os

import pandas as pd

def check_column_number(
    x:pd.DataFrame,
    y:Union[pd.DataFrame, List[pd.DataFrame]],
    x_like:List[pd.DataFrame]=[],
    y_like:List[pd.DataFrame]=[]
):
    """
    Helper function to check if x and y have the appropriate number of columns.
    If x has only one column, it will be repeated to match the number of columns in y.
    If y has only one column, it will be repeated to match the number of columns in x.
    If x and y have different number of columns, an error will be raised.
    If x_like and y_like are provided, the same checks will be performed for them (x_like elements and y_like elements must have the same number of rows as x and y, respectively).

    :param x: Input DataFrame.
    :param y: Input DataFrame.
    :param x_like: List of DataFrames.
    :param y_like: List of DataFrames.

    :return: Tuple of DataFrames.
    """

    if len(x_like) > 0:
        for x_like_ in x_like:
            assert x_like_.shape[0] == x.shape[0], "x and x_like must have the same number of rows."
            assert x_like_.shape[1] == x.shape[1], "x and x_like must have the same number of columns."

    if len(y_like) > 0:
        for y_like_ in y_like:
            y_element = y[0] if isinstance(y, list) else y

            assert y_like_.shape[0] == y_element.shape[0], "y and y_like must have the same number of rows."
            assert y_like_.shape[1] == y_element.shape[1], "y and y_like must have the same number of columns."

    x_like_output = x_like
    y_like_output = y_like

    y_element = y[0] if isinstance(y, list) else y

    if x.shape[1] != y_element.shape[1]:
        if x.shape[1] == 1:
            x = pd.concat([x]*y_element.shape[1], axis=1)
            if len(x_like) > 0:
                x_like_output = [] 
                for x_like_ in x_like:
                    x_like_output.append(pd.concat([x_like_]*x.shape[1], axis=1))

        elif isinstance(y, list):
            for i, y_element in enumerate(y):
                if y_element.shape[1] != 1:
                    raise ValueError("x and y must have the same number of columns, or one of them must have only one column.")

                else:
                    y[i] = pd.concat([y_element]*x.shape[1], axis=1)
                    if len(y_like) > 0:
                        y_like_output = []
                        for y_like_ in y_like:
                            y_like_output.append(pd.concat([y_like_]*y[i].shape[1], axis=1))

        elif y.shape[1] == 1:
            y = pd.concat([y]*x.shape[1], axis=1)
            if len(y_like) > 0:
                y_like_output = []
                for y_like_ in y_like:
                    y_like_output.append(pd.concat([y_like_]*y.shape[1], axis=1))
        else:
            raise ValueError("x and y must have the same number of columns, or one of them must have only one column.")

    return x, y, x_like_output, y_like_output


class HiddenPrints:
    """
    Context manager to temporarily suppress printing.

    Usage:
    ```python
    with HiddenPrints():
        # code with suppressed prints
    ```
    """

    def __enter__(self):
        """
        Enter the context manager and suppress printing.

        :return: None
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and restore printing.

        :return: None
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout
    
import numpy as np
import pandas as pd
from functools import partial

from sklearn.metrics import r2_score, mean_absolute_error
# import keras.backend as K
import numpy.linalg as lin
import sklearn.metrics as skm


from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(y_pred, y_test, get_cm=False):
    cm = confusion_matrix(y_pred, y_test)
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    # misclass = 1 - accuracy
    metric = {"confusion_matrix": accuracy}
    if get_cm:
        return cm
    else: 
        return metric




# @sf.node
def compute_metrics(y_pred, y_test):

    y_diff = np.abs(np.array(y_pred) - np.array(y_test)) / np.abs(np.array(y_test) + 0.01)
    y_diff_sort = np.sort(y_diff)

    quantile90, quantile95, quantile99 = np.quantile(y_diff_sort, [0.9, 0.95, 0.99])
    lstsq = lin.lstsq(y_test, y_pred)[0][0][0]
    
    #KDF Method 1
    # KDF_accuracy_ver2 = np.float64(KDF_accuracy_v2(K.constant(y_test),K.constant(y_pred)).numpy())

    skmetrics = {
        'explained_variance': skm.explained_variance_score,
        'max_error': skm.max_error,
        'mean_absolute_error': skm.mean_absolute_error,
        'mse': partial(skm.mean_squared_error, squared=True),
        'rmse': partial(skm.mean_squared_error, squared=False),
        'mean_squared_log_error': skm.mean_squared_log_error,
        'median_absolute_error': skm.median_absolute_error,
        'R2': skm.r2_score,
        'mean_poisson_deviance': skm.mean_poisson_deviance,
        'mean_gamma_deviance': skm.mean_gamma_deviance,
        'mean_absolute_percentage_error': skm.mean_absolute_percentage_error,
    }

    metrics = {
            'quantile90': quantile90,
            'quantile95': quantile95,
            'quantile99': quantile99,
            'lstsq': lstsq,
            # 'KDF_accuracy_v2': KDF_accuracy_ver2
    }


    for key, func in skmetrics.items():
        try:
            metrics[key] = func(y_test, y_pred)
            # print(key, func)
        except Exception:
            metrics[key] = None

    return metrics
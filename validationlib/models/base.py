'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
import tensorflow as tf
import numpy as np

def MSE(y: np.ndarray, a: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two arrays.

    :param y: np.ndarray
        The true values.
    :param a: np.ndarray
        The predicted values.

    :return: float
        The Mean Squared Error between y and a.
    """
    m = y.shape[0]
    return np.sum((y - a) ** 2) / m

def memoryUsageKeras(model: object, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.
    The model shapes are multipled by the batch size, but the weights are not.
    REF: https://gist.github.com/jamesmishra/34bac09176bc07b1f0c33886e4b19dc7

    Args:
        model: object
            A Keras model.
        batch_size: int
            The batch size you intend to run the model with. If you have already specified the batch size in the model itself, then pass `1` as the argument here.

    Returns:
        float
        An estimate of the Keras model's memory usage in gigabytes.
    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += memoryUsageKeras(
                layer, batch_size=batch_size
            )
        try: layer_dtype = layer.dtype
        except: layer_dtype = default_dtype

        single_layer_mem = tf.as_dtype(layer_dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory/1024**3
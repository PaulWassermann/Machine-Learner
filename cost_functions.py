import numpy as np

"""
In this file, we define classes representing cost functions to compute how well the network performed after an epoch of 
training.
"""


class MeanSquaredError:

    @staticmethod
    def compute(x, y) -> float:

        if x.ndim == 2:
            x = x[np.newaxis, ...]
            y = y[np.newaxis, ...]

        n = x.shape[0]

        return np.sum(np.sum(np.square(y - x), axis=1)) / (2 * n)

    @staticmethod
    def compute_derivative(x: np.array, y: np.array) -> np.array:

        if x.ndim == 2:
            x = x[np.newaxis, ...]
            y = y[np.newaxis, ...]

        n = x.shape[0]

        return np.sum(x - y, axis=0) / n


import numpy as np
from scipy.special import expit

"""
In this file, we define classes representing activation functions to be applied between consecutive layers.
"""


class ReLU:

    name = "relu"

    @staticmethod
    def compute(x: np.array) -> np.array:
        return np.maximum(x, np.zeros(x.shape))

    @staticmethod
    def compute_derivative(x: np.array) -> np.array:
        return np.heaviside(x, np.zeros(x.shape))


class Sigmoid:

    name = "sigmoid"

    @staticmethod
    def compute(x: np.array) -> np.array:
        return expit(x)

    @staticmethod
    def compute_derivative(x: np.array) -> np.array:
        return Sigmoid.compute(x) * (1 - Sigmoid.compute(x))


class HyperbolicTangent:

    name = "hyperbolic tangent"

    @staticmethod
    def compute(x: np.array) -> np.array:
        return np.tanh(x)

    @staticmethod
    def compute_derivative(x: np.array) -> np.array:
        return 1 - np.square(np.tanh(x))


class Softmax:

    name = "softmax"

    @staticmethod
    def compute(x: np.array) -> np.array:

        # We perform a shift in the values to prevent unstable behavior due to the use of np.exp
        maxima = np.max(x, axis=1, keepdims=True)
        x -= maxima

        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    @staticmethod
    def compute_derivative(x: np.array) -> np.array:
        return np.apply_along_axis(lambda x_i: (np.diagflat(x_i) - np.dot(x_i, x_i.T)), axis=0, arr=x)


activation_functions = {
    ReLU.name: ReLU,
    Sigmoid.name: Sigmoid,
    HyperbolicTangent.name: HyperbolicTangent,
    Softmax.name: Softmax
}

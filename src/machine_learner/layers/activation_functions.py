import numpy as np
from typing import Union
from scipy.special import expit
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from ..extras.factory import BaseFactory
from ..type_stubs import Number, npNumber


class ActivationFunction(ABC):

    name = "base activation function"

    @staticmethod
    @abstractmethod
    def compute(x: NDArray[npNumber]) -> Union[Number, NDArray[npNumber]]:
        ...

    @staticmethod
    @abstractmethod
    def compute_derivative(x: NDArray[npNumber]) -> Union[Number, NDArray[npNumber]]:
        ...


class ReLU(ActivationFunction):

    name = "relu"

    @staticmethod
    def compute(x: NDArray[npNumber]) -> NDArray[npNumber]:
        return np.maximum(x, np.zeros(x.shape))

    @staticmethod
    def compute_derivative(x: NDArray[npNumber]) -> NDArray[npNumber]:
        return np.heaviside(x, np.zeros(x.shape))


class Sigmoid(ActivationFunction):

    name = "sigmoid"

    @staticmethod
    def compute(x: NDArray[npNumber]) -> Union[Number, NDArray[npNumber]]:
        return expit(x)

    @staticmethod
    def compute_derivative(x: NDArray[npNumber]) -> Union[Number, NDArray[npNumber]]:
        return Sigmoid.compute(x) * (1 - Sigmoid.compute(x))


class HyperbolicTangent(ActivationFunction):

    name = "hyperbolic tangent"

    @staticmethod
    def compute(x: NDArray[npNumber]) -> NDArray[npNumber]:
        return np.tanh(x)

    @staticmethod
    def compute_derivative(x: NDArray[npNumber]) -> NDArray[npNumber]:
        return 1 - np.square(np.tanh(x))


class Softmax(ActivationFunction):

    name = "softmax"

    @staticmethod
    def compute(x: NDArray[npNumber]) -> NDArray[npNumber]:

        if x.ndim == 2:
            x = x[np.newaxis, ...]

        # We perform a shift in the values to prevent unstable behavior due to the use of np.exp
        maxima = np.max(x, axis=1, keepdims=True)

        x -= maxima

        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    @staticmethod
    def compute_derivative(x: NDArray[npNumber]) -> NDArray[npNumber]:

        return np.apply_along_axis(lambda x_i: (np.diagflat(x_i) - x_i * x_i.T), axis=0, arr=Softmax.compute(x))


class ActivationFunctionFactory(BaseFactory):

    _instance = "activation function"

    _map = {
        ReLU.name: ReLU,
        Sigmoid.name: Sigmoid,
        HyperbolicTangent.name: HyperbolicTangent,
        Softmax.name: Softmax
    }

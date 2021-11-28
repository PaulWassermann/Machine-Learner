import numpy as np
from typing import Union
from .type_stubs import Number
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from .extras.factory import BaseFactory


class LossFunction(ABC):

    @staticmethod
    @abstractmethod
    def compute(
            x: NDArray[Number],
            y: NDArray[Number]
            ) -> Union[Number, NDArray[Number]]:
        ...

    @staticmethod
    @abstractmethod
    def compute_derivative(
            x: NDArray[Number],
            y: NDArray[Number]
            ) -> NDArray[Number]:
        ...


class MeanSquaredError(LossFunction):

    @staticmethod
    def compute(x: NDArray[Number], y: NDArray[Number]) -> Number:

        if x.ndim == 2:
            x = x[np.newaxis, ...]
            y = y[np.newaxis, ...]

        return np.mean(np.sum(np.square(y - x), axis=1)) / 2

    @staticmethod
    def compute_derivative(x: NDArray[Number], y: NDArray[Number]) -> Union[Number, NDArray[Number]]:

        if x.ndim == 2:
            x = x[np.newaxis, ...]
            y = y[np.newaxis, ...]

        return np.mean(x - y, axis=0)


class CrossEntropy(LossFunction):

    @staticmethod
    def compute(x: NDArray[Number], y: NDArray[Number]) -> Union[Number, NDArray[Number]]:

        if x.ndim == 2:
            x = x[np.newaxis, ...]
            y = y[np.newaxis, ...]

        epsilon = 1e-8

        return - np.mean(np.sum(y * np.log(x + epsilon)))

    @staticmethod
    def compute_derivative(x: NDArray[Number], y: NDArray[Number]) -> NDArray[Number]:

        if x.ndim == 2:
            x = x[np.newaxis, ...]
            y = y[np.newaxis, ...]

        return np.mean(x - y, axis=0)


class LossFunctionFactory(BaseFactory):

    _instance = "loss function"

    _map = {
        "mse": MeanSquaredError,
        "cross_entropy": CrossEntropy
    }

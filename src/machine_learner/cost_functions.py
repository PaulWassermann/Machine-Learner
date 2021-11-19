import numpy as np
from typing import Union
from .type_stubs import Number
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class CostFunction(ABC):

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


class MeanSquaredError(CostFunction):

    @staticmethod
    def compute(x: NDArray[Number], y: NDArray[Number]) -> Number:

        if x.ndim == 2:
            x = x[np.newaxis, ...]
            y = y[np.newaxis, ...]

        n = x.shape[0]

        return np.sum(np.sum(np.square(y - x), axis=1)) / (2 * n)

    @staticmethod
    def compute_derivative(x: NDArray[Number], y: NDArray[Number]) -> Union[Number, NDArray[Number]]:

        if x.ndim == 2:
            x = x[np.newaxis, ...]
            y = y[np.newaxis, ...]

        n = x.shape[0]

        return np.sum(x - y, axis=0) / n

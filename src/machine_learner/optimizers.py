from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from .extras.factory import BaseFactory
from .type_stubs import Number, npNumber


class Optimizer(ABC):

    name = "base optimizer"

    def __init__(self, learning_rate: Number, regularization_rate: Number) -> None:
        self.learning_rate: Number = learning_rate
        self.regularization_rate: Number = regularization_rate
        self.epsilon: Number = 1e-8
        self.momentum_coefficient: Number = 0.5
        self.data: dict = {}

    @abstractmethod
    def optimize(self, a: NDArray[npNumber], b: NDArray[npNumber], value: str) -> NDArray[npNumber]:
        ...


class SGD(Optimizer):

    name = "sgd"

    def __init__(self,
                 learning_rate: Number = 0.01,
                 regularization_rate: Number = 1e-4) -> None:
        super().__init__(learning_rate, regularization_rate)

    def optimize(self,
                 x: NDArray[npNumber],
                 loss_gradient: NDArray[npNumber],
                 value: str) -> NDArray[npNumber]:
        """
        This method performs a regular gradient descent.

        :param x: an array of arrays that is to be optimized (the weight matrix, the bias vector, ...)
        :param loss_gradient: the gradient of the neural network loss function with respect to the w parameter
        :param value: the name of the parameter to be optimized
        :return: the updated value of the x parameter
        """

        return (1 - self.learning_rate * self.regularization_rate) * x - self.learning_rate * loss_gradient


class AdaGrad(Optimizer):

    name = "adagrad"

    def __init__(self,
                 learning_rate: Number = 0.01,
                 regularization_rate: Number = 1e-4) -> None:
        super().__init__(learning_rate, regularization_rate)

    def optimize(self,
                 x: NDArray[npNumber],
                 loss_gradient: NDArray[npNumber],
                 value: str) -> NDArray[npNumber]:
        """
        This method computes an adapted learning rate based on the value of previous gradient values.

        :param x: an array of arrays that is to be optimized (the weight matrix, the bias vector, ...)
        :param loss_gradient: the gradient of the neural network loss function with respect to the w parameter
        :param value: the name of the value to be optimized
        :return: the updated value of the x parameter
        """

        accumulated_gradients = self.data.get(value, np.zeros(loss_gradient.shape))

        accumulated_gradients += loss_gradient ** 2

        self.data[value] = accumulated_gradients

        adjusted_learning_rate = self.learning_rate / np.sqrt(accumulated_gradients + self.epsilon)

        return (1 - self.learning_rate * self.regularization_rate) * x - adjusted_learning_rate * loss_gradient


class MomentumSGD(Optimizer):

    name = "momentum_sgd"

    def __init__(self,
                 learning_rate: Number = 0.01,
                 regularization_rate: Number = 1e-4) -> None:
        super().__init__(learning_rate, regularization_rate)

    def optimize(self,
                 x: NDArray[npNumber],
                 loss_gradient: NDArray[npNumber],
                 value: str) -> NDArray[npNumber]:
        """
        This method performs a gradient descent smoothed by a momentum term.

        :param x: an array of arrays that is to be optimized (the weight matrix, the bias vector, ...)
        :param loss_gradient: the gradient of the neural network loss function with respect to the w parameter
        :param value: the name of the value to be optimized
        :return: the updated value of the x parameter
        """

        momentum = self.momentum_coefficient * self.data.get(value, 0) - self.learning_rate * loss_gradient

        self.data[value] = momentum

        return (1 - self.learning_rate * self.regularization_rate) * x + momentum


class OptimizerFactory(BaseFactory):

    _instance = "optimizer"

    _map = {
        SGD.name: SGD,
        AdaGrad.name: AdaGrad,
        MomentumSGD.name: MomentumSGD
    }

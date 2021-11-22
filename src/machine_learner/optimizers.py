from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from .type_stubs import Number, npNumber


class Optimizer(ABC):

    name = "base optimizer"

    def __init__(self,
                 learning_rate: Number,
                 regularization_rate: Number) -> None:
        self.learning_rate: Number = learning_rate
        self.regularization_rate: Number = regularization_rate
        self.epsilon: Number = 1e-8
        self.momentum_coefficient: Number = 0.5

    @abstractmethod
    def optimize(self,
                 a: NDArray[npNumber],
                 b: NDArray[npNumber],
                 **kwargs) -> NDArray[npNumber]:
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
                 **kwargs) -> NDArray[npNumber]:
        """
        This method performs a regular gradient descent.

        :param x: an array of arrays that is to be optimized (the weight matrix, the bias vector, ...)
        :param loss_gradient: the gradient of the neural network loss function with respect to the w parameter
        :param kwargs: no use
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
                 **kwargs) -> NDArray[npNumber]:
        """
        This method computes an adapted learning rate based on the value of previous gradient values.

        :param x: an array of arrays that is to be optimized (the weight matrix, the bias vector, ...)
        :param loss_gradient: the gradient of the neural network loss function with respect to the w parameter
        :param kwargs:
        - accumulated_gradients: an array the same shape as loss_gradient containing squared gradients from previous
        time steps
        :return: the updated value of the x parameter
        """

        accumulated_gradients = kwargs.get("accumulated_gradients")

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
                 **kwargs) -> NDArray[npNumber]:
        """
        This method computes an adapted learning rate based on the value of previous gradient values.

        :param x: an array of arrays that is to be optimized (the weight matrix, the bias vector, ...)
        :param loss_gradient: the gradient of the neural network loss function with respect to the w parameter
        :param kwargs: accepted keys are:
        - momentum: a
        :return: the updated value of the x parameter
        """

        momentum = kwargs.get("momentum", 0)

        momentum = self.momentum_coefficient * momentum - self.learning_rate * loss_gradient

        return (1 - self.learning_rate * self.regularization_rate) * x + momentum, momentum


optimizers = {
    SGD.name: SGD,
    AdaGrad.name: AdaGrad,
    MomentumSGD.name: MomentumSGD
}

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

    @abstractmethod
    def optimize(self,
                 a: NDArray[npNumber],
                 b: NDArray[npNumber]) -> NDArray[npNumber]:
        ...


class SGD(Optimizer):

    name = "sgd"

    def __init__(self,
                 learning_rate: Number = 0.01,
                 regularization_rate: Number = 1e-4) -> None:
        super().__init__(learning_rate, regularization_rate)

    def optimize(self,
                 x: NDArray[npNumber],
                 loss_gradient: NDArray[npNumber]) -> NDArray[npNumber]:
        """
        This method performs a regular gradient descent.

        :param x: an array of arrays that is to be optimized (the weight matrix, the bias vector, ...)
        :param loss_gradient: the gradient of the neural network loss function with respect to the w parameter
        :return: the updated value of the x parameter
        """

        return (1 - self.learning_rate * self.regularization_rate) * x - self.learning_rate * loss_gradient


class AdaGrad(Optimizer):

    name = "adagrad"

    def __init__(self,
                 learning_rate: Number = 0.5,
                 regularization_rate: Number = 1e-4) -> None:
        super().__init__(learning_rate, regularization_rate)

    def optimize(self,
                 x: NDArray[npNumber],
                 accumulated_gradients: NDArray[npNumber]) -> NDArray[npNumber]:
        """
        This method computes an adapted learning rate based on the value of previous gradient values.

        :param x: an array of arrays that is to be optimized (the weight matrix, the bias vector, ...)
        :param accumulated_gradients: an array containing the gradients computed at previous timesteps
        :return: the updated value of the x parameter
        """

        gradient = accumulated_gradients[-1]

        g = np.sum(accumulated_gradients ** 2, axis=0)

        diag_g = np.diag(g)[:, np.newaxis]

        return (1 - self.learning_rate * self.regularization_rate) * x \
            - self.learning_rate * gradient / (np.sqrt(diag_g + self.epsilon))


optimizers = {
    SGD.name: SGD,
    AdaGrad.name: AdaGrad
}

from abc import ABC, abstractmethod
from cost_functions import MeanSquaredError

import numpy as np


class Optimizer(ABC):

    def __init__(self, learning_rate=0.01, regularization_rate=1e-4):
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.epsilon = 1e-8

    @abstractmethod
    def optimize(self, a: np.array, b: np.array) -> np.array:
        ...


class SGD(Optimizer):

    name = "sgd"

    def optimize(self, x: np.array, loss_gradient: np.array) -> np.array:
        """
        This method performs a regular gradient descent.

        :param x: an array of arrays that is to be optimized (the weight matrix, the bias vector, ...)
        :param loss_gradient: the gradient of the neural network loss function with respect to the w parameter
        :return: the updated value of the x parameter
        """

        return (1 - self.learning_rate * self.regularization_rate) * x - self.learning_rate * loss_gradient


class AdaGrad(Optimizer):

    name = "adagrad"

    def __init__(self, learning_rate=0.5, regularization_rate=1e-4):
        super().__init__(learning_rate, regularization_rate)

    def optimize(self, x: np.array, accumulated_gradients: np.array) -> np.array:
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

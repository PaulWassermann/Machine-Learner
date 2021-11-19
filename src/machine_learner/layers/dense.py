import math
import numpy as np
from ..layers import Layer
from ..type_stubs import Number
from numpy.typing import NDArray
from ..optimizers import Optimizer
from ..layers import ActivationFunction, activation_functions


class Dense(Layer):

    number_of_dense_layers = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        Dense.number_of_dense_layers += 1
        self.name: str = f"Dense_{Dense.number_of_dense_layers}"

        self.input_neurons: int = kwargs.get("input neurons", 0)
        self.output_neurons: int = kwargs.get("output neurons", 1)

        self.x: NDArray[Number] = np.array([])
        self.z: NDArray[Number] = np.array([])
        self.y: NDArray[Number] = np.array([])

        self.weights: NDArray[Number] = np.array([])
        self.biases: NDArray[Number] = np.array([])

        self.weights_gradient: NDArray[Number] = np.array([])
        self.biases_gradient: NDArray[Number] = np.array([])

        self.weights_gradients_acc: list[NDArray[Number]] = []
        self.biases_gradients_acc: list[NDArray[Number]] = []
        self.accumulator_size: int = 20

        self.activation_function: ActivationFunction = \
            activation_functions[kwargs.get("activation function", "sigmoid")]

        if self.input_neurons != 0:
            self.init_architecture(self.input_neurons)

    def propagate_forward(self, x: NDArray[Number]) -> NDArray[Number]:
        """
        This method computes the output of the layer for a given input.

        :param x: an array of input vectors
        :return: an array containing the output vector for each input vector
        """
        if self.input_neurons == 0:
            input_neurons = x.shape[1]
            self.init_architecture(input_neurons=input_neurons)

        self.x = x

        self.z = self.weights @ self.x + self.biases

        self.y = self.activation_function.compute(self.z)

        return self.y

    def propagate_backwards(self, err_from_next_layer: NDArray[Number]) -> NDArray[Number]:
        """
        Given the error from the next layer (one step closer to the network output), updates the weights and biases
        gradient and returns the error to be propagated in the precedent layer (one step closer to the network input).

        :param err_from_next_layer: an array containing the error computed at the next layer
        :return: an array containing the error to be propagated deeper in the network
        """

        # start = perf_counter()

        propagated_error = self.activation_function.compute_derivative(self.z) * err_from_next_layer

        self.weights_gradient = np.mean(propagated_error @ np.transpose(self.x, axes=[0, 2, 1]), axis=0)
        self.weights_gradients_acc.append(self.weights_gradient)
        # self.weights_gradients_acc = np.append(self.weights_gradients_acc, self.weights_gradient[np.newaxis, ...],
        #                                        axis=0)
        if len(self.weights_gradients_acc) > self.accumulator_size:
            self.weights_gradients_acc.pop(0)

        self.biases_gradient = np.mean(propagated_error, axis=0)
        self.biases_gradients_acc.append(self.biases_gradient)
        # self.biases_gradients_acc = np.append(self.biases_gradients_acc, self.biases_gradient[np.newaxis, ...],
        #                                       axis=0)
        if len(self.biases_gradients_acc) > self.accumulator_size:
            self.biases_gradients_acc.pop(0)

        # print(f"Backward propagation execution time: {(perf_counter() - start)*100:.2f}ms")

        return self.weights.T @ propagated_error

    def update_weights(self, optimizer: Optimizer) -> None:
        """
        Given the neural network optimizer, this method optimizes the weights parameters of this layer.

        :param optimizer: Optimizer subclass, strategy used for the neural network parameters to be updated
        :return: None
        """

        # start = perf_counter()

        if optimizer.name == "sgd":
            self.weights = optimizer.optimize(self.weights, self.weights_gradient)

        elif optimizer.name == "adagrad":
            self.weights = optimizer.optimize(self.weights, np.array(self.weights_gradients_acc))

        # print(f"Weights update execution time: {(perf_counter() - start)*100:.2f}ms")

        self.weights_gradient = np.zeros(self.weights.shape)

    def update_biases(self, optimizer: Optimizer) -> None:
        """
        Given the neural network optimizer, this method optimizes the biases parameters of this layer.

        :param optimizer: Optimizer subclass, strategy used for the neural network parameters to be updated
        :return: None
        """

        if optimizer.name == "sgd":
            self.biases = optimizer.optimize(self.biases, self.biases_gradient)

        elif optimizer.name == "adagrad":
            self.biases = optimizer.optimize(self.biases, np.array(self.biases_gradients_acc))

        self.biases_gradient = np.zeros(self.biases.shape)

    def init_architecture(self, input_neurons: int) -> None:
        """
        Initializes the parameters of this layer.

        :param input_neurons: size of the input vectors that will be passed to the layer
        :return: None
        """

        self.input_neurons = input_neurons

        self.x = np.zeros((input_neurons, 1))
        self.z = np.zeros((self.output_neurons, 1))
        self.y = np.zeros((self.output_neurons, 1))

        self.init_weights()
        self.biases = np.zeros((self.output_neurons, 1))

        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

        # self.weights_gradients_acc = np.zeros((1, *self.weights.shape))
        # self.biases_gradients_acc = np.zeros((1, *self.biases.shape))

        self.weights_gradients_acc = []
        self.biases_gradients_acc = []

    def init_weights(self) -> None:
        """
        Depending on the activation function assigned to the layer, we use different strategies to initialize
        the weights matrix.

        See: https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init for more information.

        :return: None
        """
        self.weights = np.random.sample((self.output_neurons, self.input_neurons)) * 2 - 1

        r = math.sqrt(6 / (self.input_neurons + self.output_neurons))

        if self.activation_function.name == "relu":
            self.weights *= math.sqrt(2) * math.sqrt(6 / (self.input_neurons + self.output_neurons))

        elif self.activation_function.name == "sigmoid":
            self.weights *= 4 * r

        elif self.activation_function.name == "hyperbolic tangent":
            self.weights *= r
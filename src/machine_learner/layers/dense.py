import math
import numpy as np
from ..layers import Layer
from ..type_stubs import Number
from numpy.typing import NDArray
from ..optimizers import Optimizer
from ..layers import ActivationFunction, ActivationFunctionFactory


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

        self.activation_function: type(ActivationFunction) = \
            ActivationFunctionFactory.create(kwargs.get("activation function", "sigmoid"))

        self.is_output = kwargs.get("last layer", True)

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

        if self.is_output and self.activation_function.name == "softmax":
            propagated_error = err_from_next_layer

        else:
            propagated_error = self.activation_function.compute_derivative(self.z) * err_from_next_layer

        self.weights_gradient = np.mean(propagated_error @ np.transpose(self.x, axes=[0, 2, 1]), axis=0)

        if propagated_error.ndim == 2:
            self.biases_gradient = propagated_error

        else:
            self.biases_gradient = np.mean(propagated_error, axis=0)

        return self.weights.T @ propagated_error

    def update_parameters(self, optimizer: Optimizer) -> None:

        self.update_weights(optimizer)
        self.update_biases(optimizer)

    def update_weights(self, optimizer: Optimizer) -> None:
        """
        Given the neural network optimizer, this method optimizes the weights parameters of this layer.

        :param optimizer: Optimizer subclass, strategy used for the neural network parameters to be updated
        :return: None
        """

        # start = perf_counter()

        self.weights = optimizer.optimize(self.weights, self.weights_gradient, value=f"{self.name}_weights")

        # print(f"Weights update execution time: {(perf_counter() - start)*100:.2f}ms")

        self.weights_gradient = np.zeros(self.weights.shape)

    def update_biases(self, optimizer: Optimizer) -> None:
        """
        Given the neural network optimizer, this method optimizes the biases parameters of this layer.

        :param optimizer: Optimizer subclass, strategy used for the neural network parameters to be updated
        :return: None
        """

        self.biases = optimizer.optimize(self.biases, self.biases_gradient, value=f"{self.name}_biases")

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

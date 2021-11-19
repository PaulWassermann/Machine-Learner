from machine_learner.cost_functions import MeanSquaredError
from machine_learner.optimizers import optimizers
from machine_learner.layers.dense import Dense

import numpy as np
from pickle import dump
from pathlib import Path
from time import perf_counter
from numpy.typing import NDArray
from machine_learner.utils import shuffle_together
from machine_learner.type_stubs import Number, npNumber, Architecture, Layer


class NeuralNetwork:

    def __init__(self,
                 architecture: Architecture = None,
                 optimizer: str = "sgd",
                 cost_function=MeanSquaredError) -> None:

        # ----- CLASSIFICATION ATTRIBUTES -----
        self.input = None
        self.layers: list[Layer] = []
        self.output = None
        self.classes: dict[int, str] = {}

        # ----- HYPER-PARAMETERS -----
        self.batch_size: int = 50
        self.number_of_epochs: int = 5
        self.optimizer = optimizers[optimizer]()
        self.cost_function = cost_function

        self.initialize_architecture(architecture)

    def train(self,
              x: NDArray[npNumber],
              y: NDArray[npNumber],
              validation_data: np.ndarray = None,
              validation_data_labels: np.ndarray = None,
              learning_rate: Number = None,
              **kwargs) -> None:
        """
        This method takes an input vector and a label vector.
        If no validation data is passed, the method keeps 10% of the training data to serve as validation data.

        :param x: an array of input vectors
        :param y: an array of label vectors
        :param validation_data: an array of input vectors to measure the network validation accuracy throughout the
        training session
        :param validation_data_labels: an array of label vectors to measure the network validation accuracy throughout
        the training session
        :param learning_rate: a float used to define the learning rate of the optimizer
        :param kwargs: a dictionary of optional parameters. Accepted keys are:
        - epochs: redefine the number of epochs for the network to be trained on
        - batch size: redefine the size of the batches use for training
        :return: None
        """

        # Dealing with the keyword arguments
        self.number_of_epochs = kwargs.get("epochs", self.number_of_epochs)
        self.batch_size = kwargs.get("batch_size", self.batch_size)
        validation_split = kwargs.get("validation_split", 0.1)

        # Shuffling the training data
        x, y = shuffle_together(x, y)

        # If no validation dataset is passed as an argument, we make one from the training dataset
        if validation_data is None:
            validation_data_size = x.shape[0] // int(1 / validation_split)
            validation_data = x[:validation_data_size]
            validation_data_labels = y[:validation_data_size]
            x = x[validation_data_size:]
            y = y[validation_data_size:]

        # Sets the learning of the optimizer if passed as an argument
        if learning_rate is not None:
            self.optimizer.learning_rate = learning_rate

        # Set the number of neurons in the output layer depending on the shape of the labels included in y parameter
        if self.layers[-1].output_neurons != y.shape[1]:
            self.layers[-1].output_neurons = y.shape[1]

        # Looping over the training data set
        for epoch in range(self.number_of_epochs):

            print(f"\n--- Epoch: {epoch + 1} ---\n")
            print("Progress:")

            # Generate new batches for this epoch
            # "batches" variable is a generator
            batches = self.generate_batches(x, y)
            timings = []

            for batch in range(x.shape[0] // self.batch_size):

                start = perf_counter()

                # First, we get the batches for the network to be trained on
                x_batch, y_batch = next(batches)

                # We then calculate the output of the network for the given input
                y_estimated = self.inference(x_batch)

                # We back propagate the error through the whole network
                self.backpropagation(y_estimated, y_batch)

                # Finally, we update the weights and biases of the network
                self.update_parameters()

                timings.append(perf_counter() - start)

                if (batch % max(1, ((x.shape[0] // self.batch_size) // 10))) == 0 \
                        or batch == max(1, (x.shape[0] // self.batch_size)) - 1:
                    # Monitoring training session
                    print(f"Batch {batch + 1} / {x.shape[0] // self.batch_size}, "
                          f"accuracy: {100 * self.evaluate(x, y):.2f}%, "
                          f"validation accuracy: {100 * self.evaluate(validation_data, validation_data_labels):.2f}%, "
                          f"mean execution time per batch: {1000 * np.mean(timings):.2f} ms",
                          end="\r")

            print("\n", end="")

    def generate_batches(self,
                         x: NDArray[npNumber],
                         y: NDArray[npNumber]) -> iter:
        """
        This method creates a generator of randomized batches of size batch_size.

        :param x: an array of input vectors
        :param y: an array of label vectors
        :return: a generator with shuffled data
        """

        indices = np.arange(0, x.shape[0], 1)
        np.random.shuffle(indices)

        return ((x[indices[i * self.batch_size:(i + 1) * self.batch_size]],
                 y[indices[i * self.batch_size:(i + 1) * self.batch_size]])
                for i in range(x.shape[0] // self.batch_size))

    def inference(self, x: NDArray[npNumber]) -> NDArray[npNumber]:
        """
        This method computes the output of the network given an array of vectors.

        :param x: an array of size batch_size input vectors
        :return: the array of size batch_size containing in position i the output of the input vector x[i]
        """

        x_inferred = x

        for layer in self.layers:
            x_inferred = layer.propagate_forward(x_inferred)

        return x_inferred

    def backpropagation(self,
                        y: NDArray[npNumber],
                        expected_output: NDArray[npNumber]) -> None:
        """
        This method computes the error that propagates through each layer of the network.

        :param y: an array of size batch_size containing output vectors computed through inference
        :param expected_output: an array of size batch_size containing the desired output for each input vector
        :return: None
        """

        error_to_propagate = self.cost_function.compute_derivative(y, expected_output)

        for layer in self.layers[::-1]:
            error_to_propagate = layer.propagate_backwards(error_to_propagate)

    def update_parameters(self) -> None:
        """
        This method performs the gradient descent algorithm by calling the weights and biases update method of each
        layer.

        :return: None
        """

        for layer in self.layers:
            layer.update_weights(self.optimizer)
            layer.update_biases(self.optimizer)

    def evaluate(self, x: NDArray[npNumber], y: NDArray[npNumber]) -> float:
        """
        This method gives a sense of measure of the classification accuracy of the neural network.

        :param x: an array of input vectors
        :param y: an array of label vectors
        :return: a float between 0 an 1 measuring the accuracy of the network
        """

        return np.sum(np.argmax(self.inference(x), axis=1) == np.argmax(y, axis=1)) / x.shape[0]

    def decide(self, x: NDArray[npNumber]) -> NDArray[npNumber]:
        """
        Computes the output for each input vector of the x parameter and returns what class each of the input vectors
        was assigned to only if the "classes" key was included in the architecture provided.

        :param x: an array of input vectors
        :return: an array which i-th element is the class x[i] was assigned to
        """

        if x.ndim == 2:
            x = x[np.newaxis, ...]

        y_estimation = self.inference(x)

        classification_result = np.argmax(y_estimation, axis=1)

        if self.classes is not None:
            return np.apply_along_axis(lambda i: self.classes[i[0]], axis=1, arr=classification_result)

        else:
            return classification_result

    def initialize_architecture(self,
                                architecture: Architecture) -> None:
        """
        This method is responsible for the creation of the neural network per se.

        :param architecture: a dictionary defining the architecture of the neural network. Acceptable keys are :
        - model: a list of dictionary defining sequentially each layer of the network
        - classes: a list that allows to map the index of a label vector to the actual class it encodes
                   (e.g, ["dog", "cat"])
        :return:
        """

        last_layer = {
            # "activation function": "softmax"
        }

        if (classes := architecture.get("classes", None)) is not None:
            self.classes = {index: class_ for index, class_ in enumerate(classes)}
            num_classes = len(classes)

            if num_classes == 2:
                last_layer["activation function"] = "sigmoid"

            last_layer["output neurons"] = num_classes

        for layer in architecture["model"]:

            if layer["type"] == "dense":
                self.layers.append(Dense(**layer))

        # This last layer accounts for the classification layer
        self.layers.append(Dense(**last_layer))

    def save(self, path: str) -> None:
        """
        Saves the neural network as a binary .ai file to the path indicated (relative to the "main.py" file).

        :param path: the path to the file in which the network is to be saved.
        :return: None
        """

        count = 1

        path_ = Path(__file__).parent.joinpath(path).with_suffix(".ai")

        while path_.exists():
            path_ = path_.with_stem(f"{path_}_{count}")
            count += 1

        with path_.open(mode='wb') as dump_file:
            dump(file=dump_file, obj=self)
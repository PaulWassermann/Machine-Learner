from utils import one_hot_encoding, load_network
from neural_network import NeuralNetwork
from data import data_loader

import numpy as np
import matplotlib.pyplot as plt

# Data loading and formatting
# For the "examples" arrays, each element is a 784x1 column vector representing a flattened 28x28 gray image
# For the "labels" arrays, each element is a one hot encoded 10x1 vector
training_data_zip, validation_data_zip, test_data_zip = data_loader.load_data_wrapper()

training_data = np.array(list(training_data_zip), dtype="object")

training_examples = np.apply_along_axis(lambda x: x[0], axis=1, arr=training_data)
training_labels = np.apply_along_axis(lambda x: x[1], axis=1, arr=training_data)


validation_data = np.array(list(validation_data_zip), dtype="object")

validation_examples = np.apply_along_axis(lambda x: x[0], axis=1, arr=validation_data)
validation_labels = np.apply_along_axis(lambda x: x[1], axis=1, arr=validation_data)
validation_labels = one_hot_encoding(validation_labels)


test_data = np.array(list(test_data_zip), dtype="object")

test_examples = np.apply_along_axis(lambda x: x[0], axis=1, arr=test_data)
test_labels = np.apply_along_axis(lambda x: x[1], axis=1, arr=test_data)
test_labels = one_hot_encoding(test_labels)


# A neural network architecture is defined as a dictionary, comprising a list of dictionaries, "model", where a
# dictionary represents a hidden layer and a list "classes", used to define the classes (e.g: ["car", "bike"]).
# The input and output layers do not need to be specified, they are created accordingly to the size of the inputs/labels
architecture = {
    "model": [
        {
            "type": "dense",
            "input neurons": training_examples.shape[1],
            "output neurons": 100,
            "activation function": "relu"
        },
        # {
        #     "type": "dense",
        #     "output neurons": 100,
        #     "activation function": "sigmoid"
        # }
    ],

    "classes": [i for i in range(10)]
}

# Optimizer can be set to "sgd" or "adagrad" but the AdaGrad optimizer isn't working at the moment
neural_network = NeuralNetwork(architecture=architecture, optimizer="sgd")

neural_network.train(training_examples,
                     training_labels,
                     validation_data=validation_examples[:1000],
                     validation_data_labels=validation_labels[:1000],
                     learning_rate=0.5,
                     epochs=1,
                     batch_size=5)

# This line saves the neural network as a binary file, at the indicated path
# neural_network.save(path="my_model")

# This line loads a neural network saved as a binary file, at the indicated path
# neural_network = load_network("my_model")

print(f"\nAccuracy on test dataset: {neural_network.evaluate(test_examples, test_labels) * 100:.2f}%")

# plt.imshow(np.reshape(test_examples[100], (28, 28)), cmap="gray")
#
# plt.show()
#
# print(neural_network.decide(test_examples[100]))

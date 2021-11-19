from machine_learner.neural_network import NeuralNetwork
from machine_learner.data.data_loader import load_mnist_data, load_mnist_fashion_data

import matplotlib.pyplot as plt
import numpy as np

# training_examples, training_labels, validation_examples, validation_labels, test_examples, test_labels, classes = \
#     load_mnist_data()

training_examples, training_labels, test_examples, test_labels, classes = load_mnist_fashion_data()

# A neural network architecture is defined as a dictionary, comprising a list of dictionaries, "model", where a
# dictionary represents a hidden layer and a list "classes", used to define the classes (e.g: ["car", "bike"]).
# The input and output layers do not need to be specified, they are created accordingly to the size of the inputs/labels
architecture = {
    "model": [
        {
            "type": "dense",
            "input neurons": training_examples.shape[1],
            "output neurons": 64,
            "activation function": "relu"
        },
        {
            "type": "dense",
            "output neurons": 128,
            "activation function": "sigmoid"
        },
    ],

    "classes": classes
}

# Optimizer can be set to "sgd" or "adagrad" but the AdaGrad optimizer isn't working at the moment
neural_network = NeuralNetwork(architecture=architecture, optimizer="sgd")

neural_network.train(training_examples,
                     training_labels,
                     # validation_data=validation_examples[:1000],
                     # validation_data_labels=validation_labels[:1000],
                     learning_rate=0.5,
                     epochs=2,
                     batch_size=100)

# This line saves the neural network as a binary file, at the indicated path
neural_network.save(path="my_fashion_model")

# This line loads a neural network saved as a binary file, at the indicated path
# neural_network = load_network("my_model")

print(f"\nAccuracy on test dataset: {neural_network.evaluate(test_examples, test_labels) * 100:.2f}%")

plt.imshow(np.reshape(test_examples[10], (28, 28)), cmap="gray")

plt.show()

print(neural_network.decide(test_examples[10]))

![Tests](https://github.com/PaulWassermann/Machine-Learner/actions/workflows/tests.yml/badge.svg)

# Machine-Learner

This Python module aims to implement from scratch well-known machine learning concepts and techniques. 
The machine_learner module can be used to create models which can perform classification tasks.
 

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Requirements](#requirements)  
* [Setup](#setup)
* [Usage](#usage)
* [Features](#features)


## General info

As the name of the module indicates, it serves as learning tool for me, both regarding the implementation and the 
experiments with its functionalities.

## Technologies

The module is developed with Python 3.10 and mainly revolves around the numpy module.

It also uses matplotlib 3.5.0 for plots.

## Requirements

You'll need python 3.9 or newer versions of Python to use the package.

## Setup

Git clone or download the repository, and place the directory inside your project directory.

Then write in the command line 

``pip install -e <path>`` 

with ``<path>`` being the path to the downloaded repository relative to your project directory. 
This command will install the machine_learner package on your Python interpreter in editable mode,
meaning you can modify its source code and use it directly as such.

Et voilà, you can now ``import machine_learner`` in any of your Python projects. 

## Usage

An almost-minimal working example:

```python
from machine_learner.neural_network import NeuralNetwork
from machine_learner.data.data_loader import load_mnist_data

training_examples, training_labels, validation_examples, \
validation_labels, test_examples, test_labels, classes = load_mnist_data()

architecture = {
    "model": [
        {
            "type": "dense",
            "input neurons": training_examples.shape[1],
            "output neurons": 128,
            "activation function": "relu"
        },
    ],

    "classes": classes
}

# Creates the NeuralNetwork instance
neural_network = NeuralNetwork(architecture)

# Train the network
neural_network.train(training_examples, training_labels)

# Test the model on the test dataset
print("\nAccuracy on test dataset:" 
      f"{neural_network.evaluate(test_examples, test_labels) * 100:.2f}%")
```

## Features
<ul>
<li>Define easily a sequential architecture</li>
</br>
Note: since the code runs on CPU, I only implemented fully-connected layers 
and I advise not using more than 3 layers (output layer excluded)
 
```python
# This model defines a 3 layer architecture
# The output layer is automatically generated with a softmax activation function
# if the classification task involves 3 or more classes, else a sigmoid activation function
model = [
    {
        "type": "dense",
        "input neurons": input_example_size,
        "output neurons": 128,
        "activation function": "relu"
    },
    {
        "type": "dense",
        "output neurons": 256,
        "activation function": "sigmoid"
    }
]

# Be careful, the index of each class must be consistent with the label vectors
classes = ["dog", "cat"]

# Definition of the architecture dictionary
architecture = {
    "model": model,
    "classes": classes
}
```

<li>Choose an otpimizer as well as a loss function</li>
</br>
You can choose between two optimizers:

<ul>
<li>Stochastic Gradient Descent</li>
<li>Adaptative Gradient Descent</li>
</ul>
</br>
and two loss functions:
<ul>
<li>Mean Squared Error</li>
<li>Cross-Entropy</li>
</ul>

```python
neural_network = NeuralNetwork(architecture, 
                               optimizer="sgd", 
                               loss_function="mse")

neural_network2 = NeuralNetwork(architecture, 
                                optimizer="adagrad", 
                                loss_function="cross_entropy")
```

<li>Customize the training of the neural network</li>
</br>
If you have a validation dataset, you can provide it to the network when calling 
its ``train`` method. Else, the training dataset will be automatically split to build a 
validation dataset.

You can fix the number of epochs, the batch size, the learning rate.

Finally, you can specify if you want to monitor the training session through plots. If you choose so, you 
won't be able to close the figures without exiting the program. 

```python
neural_network.train(training_examples,
                     training_labels,
                     validation_data=validation_examples,
                     validation_data_labels=validation_labels,
                     learning_rate=0.01,
                     epochs=5,
                     batch_size=32,
                     plot_session=True)
```

<li>You can save and load trained models </li>
</br>
You can save and load networks to and from binary files.

Note: be careful when using the load function, it can load any binary file, ensure that the
file comes from a trusted source.

```python
from machine_learner.utils import load_network

neural_network.save("my_model")

neural_network = load_network("my_model")
```

</ul>

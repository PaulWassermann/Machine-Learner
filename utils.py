import numpy as np
from pickle import load
from pathlib import Path


def one_hot_encoding(x: np.array) -> np.array:
    """
    Given an array of integers, this function returns the one-hot encoding of each label as an array of arrays.

    :param x: an array of integers
    :return: an array of one-hot encoded arrays
    """

    one_hot_encoded = np.zeros((x.size, x.max() + 1))
    one_hot_encoded[np.arange(x.size), x] = 1
    return one_hot_encoded[..., np.newaxis]


def load_network(path: str):
    """
    Given a path relative to the "main.py" file, this function loads and returns a neural network from the file found at
    "path" as a NeuralNetwork instance.

    :param path: path to the binary file where the network to be loaded is stored.
    :return: the NeuralNetwork instance stored in the file found at "path"
    """

    path = Path(__file__).parent.joinpath(path).with_suffix(".ai")

    with path.open('rb') as file:
        return load(file)

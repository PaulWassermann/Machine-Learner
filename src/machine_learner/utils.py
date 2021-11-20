import numpy as np
from pickle import load
from pathlib import Path
from typing import Union, Any
from numpy.typing import NDArray
from .type_stubs import npNumber, NeuralNetwork


def normalize(x: NDArray[npNumber]) -> NDArray[npNumber]:
    """
    This function returns an array with values between 0 and 1.

    :param x: an array which values are to be normalized
    :return: an array with values between 0 and 1
    """
    return x / x.max()


def one_hot_encoding(x: NDArray[np.int8]) -> NDArray[np.int8]:
    """
    Given an array of integers, this function returns the one-hot encoding of each label as an array of arrays.

    :param x: an array of integers
    :return: an array of one-hot encoded arrays
    """

    one_hot_encoded = np.zeros((x.size, x.max() + 1))
    one_hot_encoded[np.arange(x.size), x] = 1
    return one_hot_encoded[..., np.newaxis].astype("int8")


def shuffle_together(x: NDArray[npNumber],
                     y: NDArray[npNumber]) -> tuple[NDArray[npNumber], NDArray[npNumber]]:
    """
    This function shuffles two arrays in the same exact order.

    :param x: an array to be shuffled
    :param y: an other array to be shuffled
    :return: a couple of arrays created from input arrays but shuffled in the same manner
    """
    indices = np.arange(0, x.shape[0], 1)
    np.random.shuffle(indices)

    return x[indices], y[indices]


def load_network(path: str) -> Union[Any, NeuralNetwork]:
    """
    Given a path relative to the "main.py" file, this function loads and returns a neural network from the file found at
    "path" as a NeuralNetwork instance.

    :param path: path to the binary file where the network to be loaded is stored.
    :return: the NeuralNetwork instance stored in the file found at "path"
    """

    path_ = Path(__file__).parent.joinpath(path).with_suffix(".ai")

    with path_.open('rb') as file:
        return load(file)

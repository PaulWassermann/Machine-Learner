import gzip
import pickle
import numpy as np
from pathlib import Path
from ..utils import one_hot_encoding, normalize


def load_mnist_data():
    data_path = Path(__file__).parent.joinpath("mnist_data", "mnist.pkl.gz")

    with gzip.open(data_path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

    training_examples = np.array([np.reshape(x, (784, 1)) for x in training_data[0]])
    training_labels = one_hot_encoding(np.array(training_data[1]).astype("int8"))

    validation_examples = np.array([np.reshape(x, (784, 1)) for x in validation_data[0]])
    validation_labels = one_hot_encoding(np.array(validation_data[1]).astype("int8"))

    test_examples = np.array([np.reshape(x, (784, 1)) for x in test_data[0]])
    test_labels = one_hot_encoding(np.array(test_data[1]).astype("int8"))

    return training_examples, \
        training_labels, \
        validation_examples, \
        validation_labels, \
        test_examples, \
        test_labels, \
        [str(i) for i in range(10)]


def load_mnist_fashion_data():
    train_examples_data_path = Path(__file__).parent.joinpath("mnist_fashion_data", "train_images.gz")
    train_labels_data_path = Path(__file__).parent.joinpath("mnist_fashion_data", "train_labels.gz")
    test_examples_data_path = Path(__file__).parent.joinpath("mnist_fashion_data", "test_images.gz")
    test_labels_data_path = Path(__file__).parent.joinpath("mnist_fashion_data", "test_labels.gz")

    with gzip.open(train_labels_data_path, "rb") as file:
        # training_labels = pickle.load(file, encoding="latin1")
        training_labels = one_hot_encoding(np.frombuffer(file.read(), dtype=np.uint8, offset=8))

    with gzip.open(train_examples_data_path, "rb") as file:
        # training_examples = pickle.load(file, encoding="ASCII")
        training_examples = normalize(np.frombuffer(file.read(), dtype=np.uint8, offset=16)
                                      .reshape(len(training_labels), 784, 1))

    with gzip.open(test_labels_data_path, "rb") as file:
        # test_labels = pickle.load(file, encoding="latin1")
        test_labels = one_hot_encoding(np.frombuffer(file.read(), dtype=np.uint8, offset=8))

    with gzip.open(test_examples_data_path, "rb") as file:
        # test_examples = pickle.load(file, encoding="latin1")
        test_examples = normalize(np.frombuffer(file.read(), dtype=np.uint8, offset=16)
                                  .reshape(len(test_labels), 784, 1))

    return training_examples, \
        training_labels, \
        test_examples, \
        test_labels, \
        ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

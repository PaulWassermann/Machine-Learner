from machine_learner.utils import normalize, one_hot_encoding, shuffle_together

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from numpy.random import sample, randint


def test_normalize():
    x_1d_array = np.array([0, 254, 127])
    x_2d_array = np.array([[60], [30], [15]])
    x_random = sample((10, 784, 1)) * 255

    y_1d_array = normalize(x_1d_array)
    y_2d_array = normalize(x_2d_array)
    y_random = normalize(x_random)

    assert_equal(y_1d_array, [0, 1, 0.5])
    assert_almost_equal(y_2d_array, [[1], [0.5], [0.25]])
    assert (0 <= y_random).all() and (y_random <= 1).all()


def test_one_hot_encoding():
    x_1d_array = np.array([0, 1, 2])
    x_2d_array = np.array([[2], [1], [1]])
    x_random = randint(0, 2, (10, 1))

    y_1d_array = one_hot_encoding(x_1d_array)
    y_2d_array = one_hot_encoding(x_2d_array)
    y_random = one_hot_encoding(x_random)

    assert_equal(y_1d_array, [[[1], [0], [0]], [[0], [1], [0]], [[0], [0], [1]]])
    assert_equal(y_2d_array, [[[0], [0], [1]], [[0], [1], [0]], [[0], [1], [0]]])
    assert y_random.shape <= (10, 3, 1)


def test_shuffle_together():
    x = np.array([i for i in range(100)])
    y = np.array([i for i in range(100)])

    shuffled_x, shuffled_y = shuffle_together(x, y)

    assert (shuffled_x == shuffled_y).all()
    assert (shuffled_x != x).any()

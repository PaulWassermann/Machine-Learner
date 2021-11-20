from machine_learner.layers.activation_functions import ReLU, Sigmoid, HyperbolicTangent, Softmax

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.random import sample


def test_relu():
    x_2d_array = np.array([[8], [-1]])
    x_3d_array = np.array([[[-2], [3]], [[4], [-5]]])
    x_random = sample((10, 784, 1)) * 2 - 1
    y_2d_array = ReLU.compute(x_2d_array)
    y_3d_array = ReLU.compute(x_3d_array)
    yp_2d_array = ReLU.compute_derivative(x_2d_array)
    yp_3d_array = ReLU.compute_derivative(x_3d_array)
    y_random = ReLU.compute(x_random)

    assert_almost_equal(y_2d_array, np.array([[8], [0]]))
    assert_almost_equal(y_3d_array, np.array([[[0], [3]], [[4], [0]]]))
    assert_almost_equal(yp_2d_array, np.array([[1], [0]]))
    assert_almost_equal(yp_3d_array, np.array([[[0], [1]], [[1], [0]]]))
    assert (y_random >= 0).all()


def test_sigmoid():
    x_2d_array = np.array([[10e8], [0]])
    x_3d_array = np.array([[[-10e8], [0]]])
    x_random = sample((10, 784, 1)) * 2 - 1
    y_2d_array = Sigmoid.compute(x_2d_array)
    y_3d_array = Sigmoid.compute(x_3d_array)
    yp_2d_array = Sigmoid.compute_derivative(x_2d_array)
    yp_3d_array = Sigmoid.compute_derivative(x_3d_array)
    y_random = Sigmoid.compute(x_random)

    assert_almost_equal(y_2d_array, np.array([[1], [0.5]]))
    assert_almost_equal(y_3d_array, np.array([[[0], [0.5]]]))
    assert_almost_equal(yp_2d_array, np.array([[0], [0.25]]))
    assert_almost_equal(yp_3d_array, np.array([[[0], [0.25]]]))
    assert (0 <= y_random).all() and (y_random <= 1).all()


def test_hyperbolic_tangent():
    x_2d_array = np.array([[10e8], [0]])
    x_3d_array = np.array([[[-10e8], [0]]])
    x_random = sample((10, 784, 1)) * 2 - 1
    y_2d_array = HyperbolicTangent.compute(x_2d_array)
    y_3d_array = HyperbolicTangent.compute(x_3d_array)
    yp_2d_array = HyperbolicTangent.compute_derivative(x_2d_array)
    yp_3d_array = HyperbolicTangent.compute_derivative(x_3d_array)
    y_random = HyperbolicTangent.compute(x_random)

    assert_almost_equal(y_2d_array, np.array([[1], [0]]))
    assert_almost_equal(y_3d_array, np.array([[[-1], [0]]]))
    assert_almost_equal(yp_2d_array, np.array([[0], [1]]))
    assert_almost_equal(yp_3d_array, np.array([[[0], [1]]]))
    assert (-1 <= y_random).all() and (y_random <= 1).all()


def test_softmax():
    x_2d_array = np.array([[1], [1]])
    x_3d_array = np.array([[[0], [0]], [[1], [1]]])
    x_random = sample((10, 784, 1)) * 2 - 1
    y_2d_array = Softmax.compute(x_2d_array)
    y_3d_array = Softmax.compute(x_3d_array)
    y_random = Softmax.compute(x_random)

    assert_almost_equal(y_2d_array, np.array([[[0.5], [0.5]]]))
    assert_almost_equal(y_3d_array, np.array([[[0.5], [0.5]], [[0.5], [0.5]]]))
    assert_almost_equal(np.sum(y_random, axis=1), np.ones((10, 1)))

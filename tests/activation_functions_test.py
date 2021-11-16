from layers.activation_functions import ReLU, Sigmoid, HyperbolicTangent, Softmax

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from numpy.random import sample


class MyTestCase(unittest.TestCase):

    def test_relu(self):
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
        self.assertTrue((y_random >= 0).all())

    def test_sigmoid(self):
        pass

    def test_hyperbolic_tangent(self):
        pass

    def test_softmax(self):
        pass


if __name__ == '__main__':
    unittest.main()

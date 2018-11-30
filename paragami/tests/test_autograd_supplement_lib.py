#!/usr/bin/env python3

import autograd
import autograd.numpy as np
from autograd.test_util import check_grads
import autograd.numpy.random as npr

from paragami import autograd_supplement_lib

import unittest

npr.seed(1)


class TestAutogradSupplement(unittest.TestCase):
    def test_inv(self):
        def fun(x):
            return np.linalg.inv(x)

        D = 3
        mat = npr.randn(D, D) + np.eye(D) * 2

        check_grads(fun)(mat)

    def test_solve(self):
        def fun(x, y):
            return np.linalg.solve(x, y)

        D = 3
        mat1 = npr.randn(D, D) + np.eye(D) * 2
        mat2 = npr.randn(D, D)

        check_grads(fun)(mat1, mat2)

    def test_slogdet(self):
        def fun(x):
            sign, logdet = np.linalg.slogdet(x)
            return logdet

        D = 3
        mat = npr.randn(D, D)
        mat[0, 0] += 1  # Make sure the matrix is not symmetric

        check_grads(fun)(mat)
        check_grads(fun)(-mat)


if __name__ == '__main__':
    unittest.main()

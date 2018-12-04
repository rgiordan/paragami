#!/usr/bin/env python3

import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads
from paragami import autograd_supplement_lib
import unittest

npr.seed(1)

def rand_psd(D):
    mat = npr.randn(D, D)
    return np.dot(mat, mat.T)


class TestAutogradSupplement(unittest.TestCase):
    def test_inv(self):
        def fun(x):
            return np.linalg.inv(x)

        D = 3
        mat = npr.randn(D, D) + np.eye(D) * 2

        check_grads(fun)(mat)

    def test_inv_3d(self):
        fun = lambda x: np.linalg.inv(x)

        D = 4
        mat = npr.randn(D, D, D) + 5 * np.eye(D)
        check_grads(fun)(mat)

        mat = npr.randn(D, D, D, D) + 5 * np.eye(D)
        check_grads(fun)(mat)

    def test_slogdet(self):
        def fun(x):
            sign, logdet = np.linalg.slogdet(x)
            return logdet

        D = 6
        mat = npr.randn(D, D)
        mat[0, 1] = mat[1, 0] + 1  # Make sure the matrix is not symmetric

        check_grads(fun)(mat)
        check_grads(fun)(-mat)

    def test_slogdet_3d(self):
        fun = lambda x: np.sum(np.linalg.slogdet(x)[1])
        mat = np.concatenate(
            [(rand_psd(5) + 5 * np.eye(5))[None,...] for _ in range(3)])
        # At this time, this is not supported.
        #check_grads(fun)(mat)

        # Check that it raises an error.
        fwd_grad = autograd.make_jvp(fun, argnum=0)
        def error_fun():
            return fwd_grad(mat)(mat)
        self.assertRaises(ValueError, error_fun)

    def test_solve_arg1(self):
        D = 8
        A = npr.randn(D, D) + 10.0 * np.eye(D)
        B = npr.randn(D, D - 1)
        def fun(a): return np.linalg.solve(a, B)
        check_grads(fun)(A)

    def test_solve_arg1_1d(self):
        D = 8
        A = npr.randn(D, D) + 10.0 * np.eye(D)
        B = npr.randn(D)
        def fun(a): return np.linalg.solve(a, B)
        check_grads(fun)(A)

    def test_solve_arg2(self):
        D = 6
        A = npr.randn(D, D) + 1.0 * np.eye(D)
        B = npr.randn(D, D - 1)
        def fun(b): return np.linalg.solve(A, b)
        check_grads(fun)(B)

    def test_solve_arg1_3d(self):
        D = 4
        A = npr.randn(D + 1, D, D) + 5 * np.eye(D)
        B = npr.randn(D + 1, D)
        fun = lambda A: np.linalg.solve(A, B)
        check_grads(fun)(A)

    def test_solve_arg1_3d_3d(self):
        D = 4
        A = npr.randn(D+1, D, D) + 5 * np.eye(D)
        B = npr.randn(D+1, D, D + 2)
        fun = lambda A: np.linalg.solve(A, B)
        check_grads(fun)(A)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

#import autograd
import numpy as np
#import autograd.numpy as np
from autograd.test_util import check_grads
#import copy
#import itertools
from numpy.testing import assert_array_almost_equal
import paragami
import scipy as sp
from sksparse.cholmod import cholesky
#from test_utils import QuadraticModel
import unittest

from paragami.optimization_lib import _get_matrix_from_operator
from paragami.sparse_preconditioners_lib import _get_cholesky_sqrt_mat
from paragami.sparse_preconditioners_lib import get_sym_matrix_inv_sqrt_funcs


def assert_sp_array_almost_equal(x, y):
    x_test = x.todense() if sp.sparse.issparse(x) else x
    y_test = y.todense() if sp.sparse.issparse(y) else y
    assert_array_almost_equal(x_test, y_test)



class TestSparseMatrixTools(unittest.TestCase):
    def test_choleksy_sqrt(self):
        dim = 5
        mat = np.eye(dim)
        mat[0, 1] = 0.2
        mat[1, 0] = 0.2
        mat[0, dim - 1] = 0.1
        mat[dim - 1, 0] = 0.1

        mat_sp = sp.sparse.csc_matrix(mat)
        mat_chol = cholesky(mat_sp)

        # Make sure that we are testing the fill-reducing permutation.
        assert(not np.all(mat_chol.P() == np.arange(dim)))

        mat_sqrt = _get_cholesky_sqrt_mat(mat_chol)
        assert_sp_array_almost_equal(mat_sqrt @ mat_sqrt.T, mat_sp)

    def test_sparse_preconditioners(self):
        dim = 5
        hess = np.random.random((dim, dim))
        hess = dim * np.eye(dim) + hess @ hess.T
        sp_hess = sp.sparse.csc_matrix(hess)

        # Check that it fails with a dense matrix.
        with self.assertRaises(ValueError):
            get_sym_matrix_inv_sqrt_funcs(hess)

        sp_h_sqrt_mult, sp_h_inv_sqrt_mult = \
            get_sym_matrix_inv_sqrt_funcs(sp_hess)

        h_sqrt = _get_matrix_from_operator(sp_h_sqrt_mult, dim)
        h_inv_sqrt = _get_matrix_from_operator(sp_h_inv_sqrt_mult, dim)

        # Test with a few random vectors.
        for n in range(100):
            v = np.random.random(dim)

            assert_array_almost_equal(sp_h_inv_sqrt_mult(sp_h_sqrt_mult(v)), v)

            assert_array_almost_equal(h_sqrt @ v, sp_h_sqrt_mult(v))
            assert_array_almost_equal(h_inv_sqrt @ v, sp_h_inv_sqrt_mult(v))

            assert_array_almost_equal(h_inv_sqrt @ h_sqrt, np.eye(dim))
            assert_array_almost_equal(h_sqrt.T @ h_sqrt, hess)


if __name__ == '__main__':
    unittest.main()

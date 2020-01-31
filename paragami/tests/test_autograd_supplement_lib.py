#!/usr/bin/env python3
import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads
from paragami import autograd_supplement_lib
import autograd.scipy as sp
from numpy.testing import assert_array_almost_equal
import scipy as osp
import scipy.sparse
import unittest

npr.seed(1)

def rand_psd(D):
    mat = npr.randn(D, D)
    return np.dot(mat, mat.T)


def assert_sp_array_almost_equal(x, y):
    x_test = x.todense() if sp.sparse.issparse(x) else x
    y_test = y.todense() if sp.sparse.issparse(y) else y
    assert_array_almost_equal(x_test, y_test)


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

    def test_gammaln_functions(self):
        for x in np.linspace(2.5, 3.5, 10):
            check_grads(sp.special.digamma)(x)
            check_grads(sp.special.psi)(x)
            check_grads(sp.special.gamma)(x)
            check_grads(sp.special.gammaln)(x)
            check_grads(sp.special.rgamma)(x)
            for n in range(4):
                check_grads(lambda x: sp.special.polygamma(int(n), x))(x)


class TestSparseMatrixTools(unittest.TestCase):
    def test_get_sparse_product(self):
        z_dense = np.random.random((10, 2))
        z_mat = osp.sparse.coo_matrix(z_dense)
        self.assertTrue(osp.sparse.issparse(z_mat))

        for mu_dim in [1, 2]:
            if mu_dim == 1:
                mu = np.random.random(z_mat.shape[1])
            else:
                mu = np.random.random((z_mat.shape[1], 3))

            z_mult, zt_mult = \
                autograd_supplement_lib.get_sparse_product(z_mat)
            check_grads(z_mult, modes=['rev', 'fwd'], order=4)(mu)

            z_mult2, zt_mult2 = \
                autograd_supplement_lib.get_sparse_product(2 * z_mat)
            check_grads(z_mult2, modes=['rev', 'fwd'], order=4)(mu)

            assert_array_almost_equal(z_mult(mu), z_mat @ mu)
            assert_array_almost_equal(z_mult2(mu), 2 * z_mat @ mu)

        # Check that errors are raised if the sparse matrix is not 2d
        z_dense_3d = np.random.random((2, 2, 2))
        self.assertRaises(
            ValueError,
            lambda: autograd_supplement_lib.get_sparse_product(z_dense_3d))

        z_dense_1d = np.random.random((2, ))
        self.assertRaises(
            ValueError,
            lambda: autograd_supplement_lib.get_sparse_product(z_dense_1d))

        # Check that errors are raised if the argument is more than 2d
        z_mult, zt_mult = \
            autograd_supplement_lib.get_sparse_product(z_mat)

        mu_bad = np.random.random((z_mat.shape[1], 3, 3))
        mut_bad = np.random.random((z_mat.shape[0], 3, 3))
        self.assertRaises(ValueError, lambda: z_mult(mu_bad))
        self.assertRaises(ValueError, lambda: zt_mult(mu_bad))


    def test_get_differentiable_solver(self):
        dim = 5
        z = np.eye(dim)
        z[0, 1] = 0.2
        z[1, 0] = 0.2
        z[0, dim - 1] = 0.05
        z[dim - 1, 0] = 0.1

        z_sp = osp.sparse.csc_matrix(z)

        a = np.random.random(dim)

        # Check with simple lambda functions.
        z_solve, zt_solve = autograd_supplement_lib.get_differentiable_solver(
            lambda b: osp.sparse.linalg.spsolve(z_sp, b),
            lambda b: osp.sparse.linalg.spsolve(z_sp.T, b))

        assert_array_almost_equal(
            osp.sparse.linalg.spsolve(z_sp, a), z_solve(a))
        assert_array_almost_equal(
            osp.sparse.linalg.spsolve(z_sp.T, a), zt_solve(a))

        check_grads(z_solve)(a)
        check_grads(zt_solve)(a)

        # Check with factorized matrices.
        z_factorized = osp.sparse.linalg.factorized(z_sp)
        zt_factorized = osp.sparse.linalg.factorized(z_sp.T)
        z_solve_fac, zt_solve_fac = \
            autograd_supplement_lib.get_differentiable_solver(
                z_factorized, zt_factorized)

        assert_array_almost_equal(z_factorized(a), z_solve_fac(a))
        assert_array_almost_equal(zt_factorized(a), zt_solve_fac(a))

        check_grads(z_solve_fac)(a)
        check_grads(zt_solve_fac)(a)

        # # Test with a Cholesky decomposition.
        # z_chol = cholesky(z_sp)
        # zt_chol = cholesky(z_sp.T)
        #
        # # Make sure that we are testing the fill-reducing permutation.
        # self.assertTrue(not np.all(z_chol.P() == np.arange(dim)))
        # z_solve_chol, zt_solve_chol = \
        #     autograd_supplement_lib.get_differentiable_solver(
        #         z_chol, zt_chol)
        #
        # # TODO(why is Cholesky not working?)
        # # For some reason, ``z_chol(a)`` doesn't work unless ``a``
        # # is the same (square) dimension as ``z``.  This appears to be
        # # a problem with the Cholesky decomposition, not the sparse solver.
        # z_chol(a) # Fails
        # assert_array_almost_equal(z_chol(a), z_solve_chol(a))
        # assert_array_almost_equal(zt_chol(a), zt_solve_chol(a))
        #
        # check_grads(z_solve_chol)(a)
        # check_grads(zt_solve_chol)(a)


class TestGroupedSum(unittest.TestCase):
    def test_grouped_sum(self):
        grouped_sum = autograd_supplement_lib.grouped_sum
        n_groups = 10
        n_per_group = 2
        n_obs = n_groups * n_per_group

        groups = np.repeat(np.arange(0, n_groups), n_per_group)

        def check(x):
            if x.ndim == 1:
                assert_array_almost_equal(
                    grouped_sum(x, groups),
                    np.bincount(groups, x))
            check_grads(grouped_sum)(x, groups)
            check_grads(grouped_sum)(x, groups, num_groups=n_groups + 4)

        check(np.random.random(n_obs))
        check(np.random.random((n_obs, 3)))
        check(np.random.random((n_obs, 3, 2)))

if __name__ == '__main__':
    unittest.main()

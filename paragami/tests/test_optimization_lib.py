#!/usr/bin/env python3

import autograd
import unittest
from numpy.testing import assert_array_almost_equal

import scipy as sp
import autograd.numpy as np
from autograd.test_util import check_grads

import itertools

import paragami


# This class will be used for testing.
class QuadraticModel(object):
    def __init__(self, dim):
        # Put lower bounds so we're testing the contraining functions
        # and so that derivatives of all orders are nonzero.
        self.dim = dim
        self.theta_pattern = \
            paragami.NumericArrayPattern(shape=(dim, ), lb=-10.)
        self.lambda_pattern = \
            paragami.NumericArrayPattern(shape=(dim, ), lb=-10.0)

        vec = np.linspace(0.1, 0.3, num=dim)
        self.matrix = np.outer(vec, vec) + np.eye(dim)

        self.lam = self.get_default_lambda()

    def get_default_lambda(self):
        return np.linspace(0.5, 10.0, num=self.dim)

    def get_hyper_par_objective(self, theta, lam):
        # Only the part of the objective that dependson the hyperparameters.
        return lam @ theta

    def get_objective(self, theta, lam):
        objective = 0.5 * theta.T @ self.matrix @ theta
        shift = self.get_hyper_par_objective(theta, lam)
        return objective + shift

    # Testing functions that use the fact that the optimum has a closed form.
    def get_true_optimal_theta(self, lam):
        return -1 * np.linalg.solve(self.matrix, lam)


class HyperparameterSensitivityLinearApproximation(unittest.TestCase):
    def _test_linear_approximation(self, dim,
                                   theta_free, lambda_free,
                                   use_hessian_at_opt,
                                   use_hyper_par_objective_fun):
        model = QuadraticModel(dim=dim)

        # Sanity check that the optimum is correct.
        get_objective_flat = paragami.FlattenedFunction(
            model.get_objective, free=theta_free, argnums=0,
            patterns=model.theta_pattern)
        get_objective_for_opt = paragami.Functor(
            get_objective_flat, argnums=0)
        get_objective_for_opt.cache_args(None, model.lam)
        get_objective_for_opt_grad = autograd.grad(get_objective_for_opt)
        get_objective_for_opt_hessian = autograd.hessian(get_objective_for_opt)

        opt_output = sp.optimize.minimize(
            fun=get_objective_for_opt,
            jac=get_objective_for_opt_grad,
            x0=np.zeros(model.dim),
            method='BFGS')

        theta0 = model.get_true_optimal_theta(model.lam)
        theta_flat = model.theta_pattern.flatten(theta0, free=theta_free)
        assert_array_almost_equal(theta_flat, opt_output.x)

        # Instantiate the sensitivity object.
        if use_hessian_at_opt:
            hess0 = get_objective_for_opt_hessian(theta_flat)
        else:
            hess0 = None

        if use_hyper_par_objective_fun:
            hyper_par_objective_fun = model.get_hyper_par_objective
        else:
            hyper_par_objective_fun = None

        parametric_sens = \
            paragami.HyperparameterSensitivityLinearApproximation(
                objective_fun=model.get_objective,
                opt_par_pattern=model.theta_pattern,
                hyper_par_pattern=model.lambda_pattern,
                opt_par_folded_value=theta0,
                hyper_par_folded_value=model.lam,
                opt_par_is_free=theta_free,
                hyper_par_is_free=lambda_free,
                hessian_at_opt=hess0,
                hyper_par_objective_fun=hyper_par_objective_fun)

        epsilon = 0.01
        lambda1 = model.lam + epsilon

        # Check the optimal parameters
        pred_diff = \
            parametric_sens.predict_opt_par_from_hyper_par(lambda1) - theta0
        true_diff = model.get_true_optimal_theta(lambda1) - theta0

        if (not theta_free) and (not lambda_free):
            # The model is linear in lambda, so the prediction should be exact.
            assert_array_almost_equal(pred_diff, true_diff)
        else:
            # Check the relative error.
            error = np.abs(pred_diff - true_diff)
            tol = epsilon * np.mean(np.abs(true_diff))
            if not np.all(error < tol):
                print('Error in linear approximation: ', error, tol)
            self.assertTrue(np.all(error < tol))

        # Test the Jacobian.
        get_true_optimal_theta_lamflat = paragami.FlattenedFunction(
            model.get_true_optimal_theta, patterns=model.lambda_pattern,
            free=lambda_free, argnums=0)
        def get_true_optimal_theta_flat(lam_flat):
            theta0 = get_true_optimal_theta_lamflat(lam_flat)
            return model.theta_pattern.flatten(theta0, free=theta_free)

        get_dopt_dhyper = autograd.jacobian(get_true_optimal_theta_flat)
        lambda_flat = model.lambda_pattern.flatten(model.lam, free=lambda_free)
        assert_array_almost_equal(
            get_dopt_dhyper(lambda_flat),
            parametric_sens.get_dopt_dhyper())

    def test_quadratic_model(self):
        ft_vec = [False, True]
        dim = 3
        for (theta_free, lambda_free, use_hess, use_hyperobj) in \
            itertools.product(ft_vec, ft_vec, ft_vec, ft_vec):

            print(('theta_free: {}, lambda_free: {}, ' +
                   'use_hess: {}, use_hyperobj: {}').format(
                   theta_free, lambda_free, use_hess, use_hyperobj))
            self._test_linear_approximation(
                dim, theta_free, lambda_free,
                use_hess, use_hyperobj)

class TestPreconditionedFunction(unittest.TestCase):
    def test_preconditioned_function(self):
        model = QuadraticModel(dim=3)

        # Define a function of theta alone.
        opt_objective = paragami.Functor(model.get_objective, argnums=0)
        opt_objective.cache_args(None, model.lam)

    def _test_matrix_sqrt(self, mat):
        id_mat = np.eye(mat.shape[0])
        eig_vals = np.linalg.eigvals(mat)
        ev_min = np.min(eig_vals)
        ev_max = np.max(eig_vals)
        ev0 = ev_min + (ev_max - ev_min) / 3
        ev1 = ev_min + 2 * (ev_max - ev_min) / 3

        for test_ev_min in [None, ev0]:
            for test_ev_max in [None, ev1]:
                h_inv_sqrt, h_sqrt, h = \
                    paragami.optimization_lib._get_sym_matrix_inv_sqrt(
                        mat, test_ev_min, test_ev_max)
                assert_array_almost_equal(id_mat, h_inv_sqrt @ h_sqrt)
                assert_array_almost_equal(
                    id_mat, h_inv_sqrt @ h @ h_inv_sqrt.T)
                eig_vals_test = np.linalg.eigvals(h)
                if test_ev_min is not None:
                    self.assertTrue(np.min(eig_vals_test) >=
                                    test_ev_min - 1e-8)
                else:
                    assert_array_almost_equal(ev_min, np.min(eig_vals_test))
                if test_ev_max is not None:
                    self.assertTrue(np.max(eig_vals_test) <=
                                    test_ev_max + 1e-8)
                else:
                    assert_array_almost_equal(ev_max, np.max(eig_vals_test))

    def test_matrix_sqrt(self):
        dim = 5
        mat = dim * np.eye(dim)
        vec = np.random.random(dim)
        mat = mat + np.outer(vec, vec)
        self._test_matrix_sqrt(mat)



if __name__ == '__main__':
    unittest.main()

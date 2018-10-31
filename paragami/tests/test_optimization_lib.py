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
    def __init__(self, dim, theta_free):
        # Put lower bounds so we're testing the contraining functions
        # and so that derivatives of all orders are nonzero.
        self.dim = dim
        self.theta_free = theta_free
        self.theta_pattern = \
            paragami.NumericArrayPattern(shape=(dim, ), lb=-10.)
        self.lambda_pattern = \
            paragami.NumericArrayPattern(shape=(dim, ), lb=-2.0)

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
        theta0 = -1 * np.linalg.solve(self.matrix, lam)
        return self.theta_pattern.flatten(theta0, free=self.theta_free)


class HyperparameterSensitivityLinearApproximation(unittest.TestCase):
    def _test_linear_approximation(self, dim,
                                   theta_free, lambda_free,
                                   use_hessian_at_opt,
                                   use_hyper_par_objective_fun):
        model = QuadraticModel(dim=dim, theta_free=theta_free)

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
        assert_array_almost_equal(theta0, opt_output.x)

        theta_folded = model.theta_pattern.fold(theta0, free=theta_free)

        # Instantiate the sensitivity object.
        theta0 = model.get_true_optimal_theta(model.lam)
        if use_hessian_at_opt:
            hess0 = get_objective_for_opt_hessian(theta0)
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
                opt_par_folded_value=theta_folded,
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
                print(error, tol)
            self.assertTrue(np.all(error < tol))

        # Check the Jacobian.
        get_dinput_dhyper = autograd.jacobian(model.get_true_optimal_theta)
        assert_array_almost_equal(
            get_dinput_dhyper(model.lam),
            parametric_sens.get_dopt_dhyper())

    def test_quadratic_model(self):
        ft_vec = [False, True]
        for (theta_free, lambda_free, use_hess, use_hyperobj) in \
            itertools.product(ft_vec, ft_vec, ft_vec, ft_vec):

            print(('theta_free: {}, lambda_free: {}, ' +
                   'use_hess: {}, use_hyperobj: {}').format(
                   theta_free, lambda_free, use_hess, use_hyperobj))
            self._test_linear_approximation(
                3, theta_free, lambda_free,
                use_hess, use_hyperobj)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

import unittest
from numpy.testing import assert_array_almost_equal

import autograd.numpy as np
from autograd.test_util import check_grads

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

        self.lambda0 = self.get_default_lambda()

        self.get_opt_objective_flat = paragami.FlattenedFunction(
            self.get_opt_objective, self.theta_pattern, free=self.theta_free)
        self.get_opt_objective_flat_grad = \
            autograd.grad(self.get_opt_objective_flat)
        self.get_opt_objective_flat_hessian = \
            autograd.hessian(self.get_opt_objective_flat)

        self.get_hyper_par_objective_flat = paragami.FlattenedFunction(
            self.get_hyper_par_objective,
            patterns=[ self.theta_pattern, self.lambda_pattern ],
            free=self.theta_free)

    def get_default_lambda(self):
        return np.linspace(0.5, 10.0, num=dim)

    def get_hyper_par_objective(self, theta, lambda0):
        # Only the part of the objective that dependson the hyperparameters.
        return lambda0 @ theta

    def get_objective(self, theta, lambda0):
        objective = 0.5 * theta.T @ self.matrix @ theta
        shift = self.get_hyper_par_objective(theta, lambda0)
        return objective + shift

    def get_opt_objective(self, theta):
        return self.get_opt_objective(theta, self.lambda0)

    # Testing functions that use the fact that the optimum has a closed form.
    def get_true_optimal_theta(self, lambda0):
        theta0 = -1 * np.linalg.solve(self.matrix, lambda0)
        return self.theta_pattern.flatten(theta, free=self.theta_free)


class HyperparameterSensitivityLinearApproximation(unittest.TestCase):
    def test_linear_approximation(self, dim,
                                  theta_free, lambda_free,
                                  use_hessian_at_opt,
                                  use_hyper_par_objective_fun):
        model = QuadraticModel(dim=dim, theta_free=theta_free)

        # Sanity check that the optimum is correct.
        opt_output = sp.optimize.minimize(
            fun=model.get_opt_objective_flat,
            jac=model.get_opt_objective_flat_grad,
            x0=np.zeros(model.dim),
            method='BFGS')
        theta_opt = model.theta_pattern.fold(opt_output.x, free=True)
        theta0 = model.get_true_optimum(model.lambda0)
        np_test.assert_array_almost_equal(theta0, theta_opt)

        # Instantiate the sensitivity object.
        theta0 = model.get_true_optimum(model.lambda0)
        if use_hessian_at_opt:
            hess0 = model.get_opt_objective_flat_hessian(theta0)
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
                hyper_par_folded_value=model.lambda0,
                opt_par_is_free=theta_free,
                hyper_par_is_free=False,
                hessian_at_opt=hess0,
                hyper_par_objective_fun=hyper_par_objective_fun)

        epsilon = 0.01
        lambda1 = model.lambda0 + epsilon

        # Check the optimal parameters
        pred_diff = \
            parametric_sens.predict_opt_par_from_hyper_par(lambda1) - theta0
        true_diff = model.get_true_optimum(lamda1) - theta0

        if (not theta_free) and (not lambda_free):
            # The model is linear in lambda, so the prediction should be exact.
            assert_array_almost_equal(pred_diff, true_diff)
        else:
            # Check the relative error.
            error = np.abs(pred_diff - true_diff)
            tol = epsilon * np.mean(np.abs(true_diff))
            self.assertTrue(np.all(error < tol))

        # Check the Jacobian.
        def get_true_optimal_theta(lambda0):
            return model.get_true_optimal_theta(lambda0, theta_free)

        get_dinput_dhyper = autograd.jacobian(get_true_optimal_theta)
        np_test.assert_array_almost_equal(
            get_dinput_dhyper(model.lambda0),
            parametric_sens.get_dinput_dhyper())

        # Check that the sensitivity works when specifying
        # hyper_par_objective_fun.
        # I think it suffices to just check the derivatives.
        model.param.set_free(theta0)
        model.hyper_param.set_vector(hyper_param_val)
        parametric_sens2 = sens_lib.ParametricSensitivityLinearApproximation(
            objective_functor=model.get_objective,
            input_par=model.param,
            hyper_par=model.hyper_param,
            input_val0=theta0,
            hyper_val0=hyper_param_val,
            hyper_par_objective_functor=model.get_hyper_par_objective)

        np_test.assert_array_almost_equal(
            get_dinput_dhyper(hyper_param_val),
            parametric_sens2.get_dinput_dhyper())

    def test_quadratic_model(self):
        pass

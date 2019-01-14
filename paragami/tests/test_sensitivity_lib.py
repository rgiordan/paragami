#!/usr/bin/env python3

import autograd
import autograd.numpy as np
from copy import deepcopy
import itertools
from numpy.testing import assert_array_almost_equal
import paragami
from paragami import sensitivity_lib
import scipy as sp
from test_utils import QuadraticModel
import unittest
import warnings


class TestHessianSolver(unittest.TestCase):
    def test_solver(self):
        np.random.seed(101)
        d = 10
        h_dense = np.random.random((d, d))
        h_dense = h_dense + h_dense.T + d * np.eye(d)
        h_sparse = sp.sparse.csc_matrix(h_dense)
        v = np.random.random(d)
        h_inv_v = np.linalg.solve(h_dense, v)

        for h in [h_dense, h_sparse]:
            for method in ['factorization', 'cg']:
                h_solver = sensitivity_lib.HessianSolver(h, method)
                assert_array_almost_equal(h_solver.solve(v), h_inv_v)

        h_solver = paragami.sensitivity_lib.HessianSolver(h_dense, 'cg')
        h_solver.set_cg_options({'maxiter': 1})
        with self.assertWarns(UserWarning):
            # With only one iteration, the CG should fail and raise a warning.
            h_solver.solve(v)


class TestLinearResponseCovariances(unittest.TestCase):
    def test_lr(self):
        np.random.seed(42)

        dim = 4
        mfvb_par_pattern = paragami.PatternDict()
        mfvb_par_pattern['mean'] = paragami.NumericArrayPattern((dim, ))
        mfvb_par_pattern['var'] = paragami.NumericArrayPattern((dim, ))

        mfvb_par = mfvb_par_pattern.empty(valid=True)

        true_mean = np.arange(0, dim)
        true_cov = dim * np.eye(dim) + np.outer(true_mean, true_mean)
        true_info = np.linalg.inv(true_cov)

        def get_kl(mfvb_par):
            """
            This is :math:`KL(q(\\theta) || p(\\theta))`` where
            :math:`p(\\theta)` is normal with mean ``true_mean``
            and inverse covariance ``ture_info`` and the variational
            distribution :math:`q` is given by ``mfvb_par``.
            The result is only up to constants that do not depend on
            :math:`q`.
            """

            t_centered = mfvb_par['mean'] - true_mean
            e_log_p = -0.5 * (
                np.trace(true_info @ np.diag(mfvb_par['var'])) +
                t_centered.T @ true_info @ t_centered)
            q_ent = 0.5 * np.sum(np.log(mfvb_par['var']))

            return -1 * (q_ent + e_log_p)

        par_free = True
        init_hessian = True

        for par_free, init_hessian in \
            itertools.product([False, True], [False, True]):

            get_kl_flat = paragami.FlattenFunctionInput(
                original_fun=get_kl, patterns=mfvb_par_pattern, free=par_free)
            get_kl_flat_grad = autograd.grad(get_kl_flat, argnum=0)
            get_kl_flat_hessian = autograd.hessian(get_kl_flat, argnum=0)

            # This is the optimum.
            mfvb_par['mean'] = true_mean
            mfvb_par['var'] = 1 / np.diag(true_info)
            mfvb_par_flat = mfvb_par_pattern.flatten(mfvb_par, free=par_free)

            hess0 = get_kl_flat_hessian(mfvb_par_flat)

            # Sanity check.the optimum.
            assert_array_almost_equal(
                0., np.linalg.norm(get_kl_flat_grad(mfvb_par_flat)))

            if init_hessian:
                lr_covs = paragami.LinearResponseCovariances(
                    objective_fun=get_kl_flat,
                    opt_par_value=mfvb_par_flat,
                    validate_optimum=True,
                    hessian_at_opt=hess0,
                    grad_tol=1e-15)
            else:
                lr_covs = paragami.LinearResponseCovariances(
                    objective_fun=get_kl_flat,
                    opt_par_value=mfvb_par_flat,
                    validate_optimum=True,
                    grad_tol=1e-15)

            assert_array_almost_equal(hess0, lr_covs.get_hessian_at_opt())

            get_mean_flat = paragami.FlattenFunctionInput(
                lambda mfvb_par: mfvb_par['mean'],
                patterns=mfvb_par_pattern,
                free=par_free)
            theta_lr_cov = lr_covs.get_lr_covariance(get_mean_flat)

            # The LR covariance is exact for the multivariate normal.
            assert_array_almost_equal(true_cov, theta_lr_cov)
            moment_jac = lr_covs.get_moment_jacobian(get_mean_flat)
            assert_array_almost_equal(
                theta_lr_cov,
                lr_covs.get_lr_covariance_from_jacobians(
                    moment_jac, moment_jac))

            # Check cross-covariances.
            get_mean01_flat = paragami.FlattenFunctionInput(
                lambda mfvb_par: mfvb_par['mean'][0:2],
                patterns=mfvb_par_pattern,
                free=par_free)
            get_mean23_flat = paragami.FlattenFunctionInput(
                lambda mfvb_par: mfvb_par['mean'][2:4],
                patterns=mfvb_par_pattern,
                free=par_free)
            moment01_jac = lr_covs.get_moment_jacobian(get_mean01_flat)
            moment23_jac = lr_covs.get_moment_jacobian(get_mean23_flat)
            assert_array_almost_equal(
                theta_lr_cov[0:2, 2:4],
                lr_covs.get_lr_covariance_from_jacobians(
                    moment01_jac, moment23_jac))

            # Check that you get an error when passing in a Jacobian with the
            # wrong dimension.
            self.assertRaises(
                ValueError,
                lambda: lr_covs.get_lr_covariance_from_jacobians(
                                    moment_jac.T, moment_jac))

            self.assertRaises(
                ValueError,
                lambda: lr_covs.get_lr_covariance_from_jacobians(
                                    moment_jac, moment_jac.T))

            self.assertRaises(
                ValueError,
                lambda: lr_covs.get_lr_covariance_from_jacobians(
                                    moment_jac[:, :, None], moment_jac))
            self.assertRaises(
                ValueError,
                lambda: lr_covs.get_lr_covariance_from_jacobians(
                                    moment_jac, moment_jac[:, :, None]))


class TestHyperparameterSensitivityLinearApproximation(unittest.TestCase):
    def _test_linear_approximation(self, dim,
                                   theta_free, lambda_free,
                                   use_hessian_at_opt,
                                   use_cross_hessian_at_opt,
                                   use_hyper_par_objective_fun):
        model = QuadraticModel(dim=dim)
        lam_folded0 = deepcopy(model.lam)
        lam0 = model.lambda_pattern.flatten(lam_folded0, free=lambda_free)

        # Sanity check that the optimum is correct.
        get_objective_flat = paragami.FlattenFunctionInput(
            model.get_objective,
            free=[theta_free, lambda_free],
            argnums=[0, 1],
            patterns=[model.theta_pattern, model.lambda_pattern])
        get_objective_for_opt = lambda x: get_objective_flat(x, lam0)
        get_objective_for_opt_grad = autograd.grad(get_objective_for_opt)
        get_objective_for_opt_hessian = autograd.hessian(get_objective_for_opt)

        get_objective_for_sens_grad = \
            autograd.grad(get_objective_flat, argnum=0)
        get_objective_for_sens_cross_hess = \
            autograd.jacobian(get_objective_for_sens_grad, argnum=1)

        opt_output = sp.optimize.minimize(
            fun=get_objective_for_opt,
            jac=get_objective_for_opt_grad,
            x0=np.zeros(model.dim),
            method='BFGS')

        theta_folded_0 = model.get_true_optimal_theta(model.lam)
        theta0 = model.theta_pattern.flatten(theta_folded_0, free=theta_free)
        assert_array_almost_equal(theta0, opt_output.x)

        # Instantiate the sensitivity object.
        if use_hessian_at_opt:
            hess0 = get_objective_for_opt_hessian(theta0)
        else:
            hess0 = None

        if use_cross_hessian_at_opt:
            cross_hess0 = get_objective_for_sens_cross_hess(theta0, lam0)
        else:
            cross_hess0 = None

        if use_hyper_par_objective_fun:
            hyper_par_objective_fun = \
                paragami.FlattenFunctionInput(
                    model.get_hyper_par_objective,
                    free=[theta_free, lambda_free],
                    argnums=[0, 1],
                    patterns=[model.theta_pattern, model.lambda_pattern])
        else:
            hyper_par_objective_fun = None

        parametric_sens = \
            paragami.HyperparameterSensitivityLinearApproximation(
                objective_fun=get_objective_flat,
                opt_par_value=theta0,
                hyper_par_value=lam0,
                hessian_at_opt=hess0,
                cross_hess_at_opt=cross_hess0,
                hyper_par_objective_fun=hyper_par_objective_fun,
                validate_optimum=True)

        epsilon = 0.001
        lam1 = lam0 + epsilon
        lam_folded1 = model.lambda_pattern.fold(lam1, free=lambda_free)

        # Check the optimal parameters
        pred_diff = \
            parametric_sens.predict_opt_par_from_hyper_par(lam1) - theta0
        true_theta_folded1 = model.get_true_optimal_theta(lam_folded1)
        true_theta1 = \
            model.theta_pattern.flatten(true_theta_folded1, free=theta_free)
        true_diff = true_theta1 - theta0

        if (not theta_free) and (not lambda_free):
            # The optimum is linear in lambda, so the prediction
            # should be exact.
            assert_array_almost_equal(pred_diff, true_diff)
        else:
            # Check the relative error.
            error = np.abs(pred_diff - true_diff)
            tol = 0.01 * np.max(np.abs(true_diff))
            if not np.all(error < tol):
                print('Error in linear approximation: ',
                      error, tol, pred_diff, true_diff)
            self.assertTrue(np.all(error < tol))

        # Test the Jacobian.
        get_true_optimal_theta_lamflat = \
            paragami.FlattenFunctionInput(
                model.get_true_optimal_theta,
                patterns=model.lambda_pattern,
                free=lambda_free, argnums=0)
        def get_true_optimal_theta_flat(lam_flat):
            theta_folded = get_true_optimal_theta_lamflat(lam_flat)
            return model.theta_pattern.flatten(theta_folded, free=theta_free)

        get_dopt_dhyper = autograd.jacobian(get_true_optimal_theta_flat)
        assert_array_almost_equal(
            get_dopt_dhyper(lam0),
            parametric_sens.get_dopt_dhyper())

    def test_quadratic_model(self):
        ft_vec = [False, True]
        dim = 3
        for (theta_free, lambda_free, use_hess, use_hyperobj, use_cross_hess) in \
            itertools.product(ft_vec, ft_vec, ft_vec, ft_vec, ft_vec):

            print(('theta_free: {}, lambda_free: {}, ' +
                   'use_hess: {}, use_hyperobj: {}').format(
                   theta_free, lambda_free, use_hess, use_hyperobj))
            self._test_linear_approximation(
                dim, theta_free, lambda_free,
                use_hess, use_cross_hess, use_hyperobj)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

import autograd
import unittest
from numpy.testing import assert_array_almost_equal

import scipy as sp
import autograd.numpy as np
from autograd.test_util import check_grads

import paragami
from paragami import sensitivity_lib

from test_utils import QuadraticModel


class TestTaylorExpansion(unittest.TestCase):
    def test_everything(self):
        # TODO: split some of these out into standalone tests.

        #################################
        # Set up the ground truth.

        # Perhaps confusingly, in the notation of the
        # ParametricSensitivityTaylorExpansionForwardDiff
        # and QuadraticModel class respectively,
        # eta = flattened theta
        # epsilon = flattened lambda.
        model = QuadraticModel(dim=3)

        eta_is_free = True
        eps_is_free = True
        eta0 = model.theta_pattern.flatten(
            model.get_true_optimal_theta(model.lam), free=eta_is_free)
        eps0 = model.lambda_pattern.flatten(model.lam, free=eps_is_free)

        objective = paragami.FlattenedFunction(
            original_fun=model.get_objective,
            patterns=[model.theta_pattern, model.lambda_pattern],
            free=[eta_is_free, eps_is_free],
            argnums=[0, 1])

        obj_eta_grad = autograd.grad(objective, argnum=0)
        obj_eps_grad = autograd.grad(objective, argnum=1)
        obj_eta_hessian = autograd.hessian(objective, argnum=0)
        obj_eps_hessian = autograd.hessian(objective, argnum=1)
        get_dobj_deta_deps = autograd.jacobian(
            autograd.jacobian(objective, argnum=0), argnum=1)

        hess0 = obj_eta_hessian(eta0, eps0)

        eps1 = eps0 + 1e-1
        eta1 = model.get_true_optimal_theta(eps1)

        # Get the exact derivatives using the closed-form optimum.
        def get_true_optimal_flat_theta(lam):
            theta = model.get_true_optimal_theta(lam)
            return model.theta_pattern.flatten(theta, free=eta_is_free)

        get_true_optimal_flat_theta = paragami.FlattenedFunction(
            original_fun=get_true_optimal_flat_theta,
            patterns=model.lambda_pattern,
            free=eps_is_free,
            argnums=0)
        true_deta_deps = autograd.jacobian(get_true_optimal_flat_theta)
        true_d2eta_deps2 = autograd.jacobian(true_deta_deps)
        true_d3eta_deps3 = autograd.jacobian(true_d2eta_deps2)
        true_d4eta_deps4 = autograd.jacobian(true_d3eta_deps3)

        # Sanity check using standard first-order approximation.
        d2f_deta_deps = get_dobj_deta_deps(eta0, eps0)
        assert_array_almost_equal(
            true_deta_deps(eps0),
            -1 * np.linalg.solve(hess0, d2f_deta_deps))

        ########################
        # Test append_jvp.
        dobj_deta = sensitivity_lib._append_jvp(
            objective, num_base_args=2, argnum=0)
        d2obj_deta_deta = sensitivity_lib._append_jvp(
            dobj_deta, num_base_args=2, argnum=0)

        v1 = np.random.random(len(eta0))
        v2 = np.random.random(len(eta0))
        v3 = np.random.random(len(eta0))
        w1 = np.random.random(len(eps0))
        w2 = np.random.random(len(eps0))
        w3 = np.random.random(len(eps0))

        # Check the first argument
        assert_array_almost_equal(
            np.einsum('i,i', obj_eta_grad(eta0, eps0), v1),
            dobj_deta(eta0, eps0, v1))
        assert_array_almost_equal(
            np.einsum('ij,i,j', obj_eta_hessian(eta0, eps0), v1, v2),
            d2obj_deta_deta(eta0, eps0, v1, v2))

        # Check the second argument
        dobj_deps = sensitivity_lib._append_jvp(
            objective, num_base_args=2, argnum=1)
        d2obj_deps_deps = sensitivity_lib._append_jvp(
            dobj_deps, num_base_args=2, argnum=1)

        assert_array_almost_equal(
            np.einsum('i,i', obj_eps_grad(eta0, eps0), w1),
            dobj_deps(eta0, eps0, w1))

        assert_array_almost_equal(
            np.einsum('ij,i,j', obj_eps_hessian(eta0, eps0), w1, w2),
            d2obj_deps_deps(eta0, eps0, w1, w2))

        # Check mixed arguments
        d2obj_deps_deta = sensitivity_lib._append_jvp(
            dobj_deps, num_base_args=2, argnum=0)
        d2obj_deta_deps = sensitivity_lib._append_jvp(
            dobj_deta, num_base_args=2, argnum=1)

        assert_array_almost_equal(
            d2obj_deps_deta(eta0, eps0, v1, w1),
            d2obj_deta_deps(eta0, eps0, w1, v1))

        assert_array_almost_equal(
            np.einsum('ij,i,j', get_dobj_deta_deps(eta0, eps0), v1, w1),
            d2obj_deps_deta(eta0, eps0, v1, w1))

        # Check derivatives of vectors.
        dg_deta = sensitivity_lib._append_jvp(
            obj_eta_grad, num_base_args=2, argnum=0)

        assert_array_almost_equal(
            hess0 @ v1, dg_deta(eta0, eps0, v1))

        ########################
        # Test derivative terms.

        # Again, first some ground truth.
        def eval_deta_deps(eta, eps, v1):
            assert np.max(np.sum(eps - eps0)) < 1e-8
            assert np.max(np.sum(eta - eta0)) < 1e-8
            return -1 * np.linalg.solve(hess0, d2f_deta_deps @ v1)

        dg_deta = sensitivity_lib._append_jvp(
            obj_eta_grad, num_base_args=2, argnum=0)
        dg_deps = sensitivity_lib._append_jvp(
            obj_eta_grad, num_base_args=2, argnum=1)

        d2g_deta_deta = sensitivity_lib._append_jvp(
            dg_deta, num_base_args=2, argnum=0)
        d2g_deta_deps = sensitivity_lib._append_jvp(
            dg_deta, num_base_args=2, argnum=1)
        d2g_deps_deta = sensitivity_lib._append_jvp(
            dg_deps, num_base_args=2, argnum=0)
        d2g_deps_deps = sensitivity_lib._append_jvp(
            dg_deps, num_base_args=2, argnum=1)

        # This is a manual version of the second derivative.
        def eval_d2eta_deps2(eta, eps, delta_eps):
            assert np.max(np.sum(eps - eps0)) < 1e-8
            assert np.max(np.sum(eta - eta0)) < 1e-8

            deta_deps = -1 * np.linalg.solve(
                hess0, dg_deps(eta, eps, delta_eps))

            # Then the terms in the second derivative.
            d2_terms = \
                d2g_deps_deps(eta, eps, delta_eps, delta_eps) + \
                d2g_deps_deta(eta, eps, delta_eps, deta_deps) + \
                d2g_deta_deps(eta, eps, deta_deps, delta_eps) + \
                d2g_deta_deta(eta, eps, deta_deps, deta_deps)
            d2eta_deps2 = -1 * np.linalg.solve(hess0, d2_terms)
            return d2eta_deps2

        eval_g_derivs = sensitivity_lib._generate_two_term_derivative_array(
            obj_eta_grad, order=5)

        assert_array_almost_equal(
            hess0 @ v1,
            eval_g_derivs[1][0](eta0, eps0, v1))

        d2g_deta_deta(eta0, eps0, v1, v2)
        eval_g_derivs[2][0](eta0, eps0, v1, v2)

        assert_array_almost_equal(
            d2g_deta_deta(eta0, eps0, v1, v2),
            eval_g_derivs[2][0](eta0, eps0, v1, v2))

        assert_array_almost_equal(
            d2g_deta_deps(eta0, eps0, v1, v2),
            eval_g_derivs[1][1](eta0, eps0, v1, v2))

        # Test the DerivativeTerm.

        dterm = sensitivity_lib.DerivativeTerm(
            eps_order=1,
            eta_orders=[1, 0],
            prefactor=1.5,
            eval_eta_derivs=[ eval_deta_deps ],
            eval_g_derivs=eval_g_derivs)

        deps = eps1 - eps0

        assert_array_almost_equal(
            dterm.prefactor * d2g_deta_deps(
                eta0, eps0, eval_deta_deps(eta0, eps0, deps), deps),
            dterm.evaluate(eta0, eps0, deps))

        dterms = [
            sensitivity_lib.DerivativeTerm(
                eps_order=2,
                eta_orders=[0, 0],
                prefactor=1.5,
                eval_eta_derivs=[ eval_deta_deps ],
                eval_g_derivs=eval_g_derivs),
            sensitivity_lib.DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=2,
                eval_eta_derivs=[ eval_deta_deps ],
                eval_g_derivs=eval_g_derivs),
            sensitivity_lib.DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=3,
                eval_eta_derivs=[ eval_deta_deps ],
                eval_g_derivs=eval_g_derivs) ]


        dterms_combined = sensitivity_lib._consolidate_terms(dterms)
        self.assertEqual(3, len(dterms))
        self.assertEqual(2, len(dterms_combined))

        assert_array_almost_equal(
            sensitivity_lib.evaluate_terms(dterms, eta0, eps0, deps),
            sensitivity_lib.evaluate_terms(dterms_combined, eta0, eps0, deps))

        dterms1 = sensitivity_lib._get_taylor_base_terms(eval_g_derivs)

        assert_array_almost_equal(
            dg_deps(eta0, eps0, deps),
            dterms1[0].evaluate(eta0, eps0, deps))

        assert_array_almost_equal(
            np.einsum('ij,j', true_deta_deps(eps0), deps),
            sensitivity_lib.evaluate_dketa_depsk(hess0, dterms1, eta0, eps0, deps))

        assert_array_almost_equal(
            eval_deta_deps(eta0, eps0, deps),
            sensitivity_lib.evaluate_dketa_depsk(hess0, dterms1, eta0, eps0, deps))

        dterms2 = sensitivity_lib.differentiate_terms(hess0, dterms1)
        assert np.linalg.norm(sensitivity_lib.evaluate_dketa_depsk(hess0, dterms2, eta0, eps0, deps)) > 0
        assert_array_almost_equal(
            np.einsum('ijk,j, k', true_d2eta_deps2(eps0), deps, deps),
            sensitivity_lib.evaluate_dketa_depsk(hess0, dterms2, eta0, eps0, deps))

        dterms3 = sensitivity_lib.differentiate_terms(hess0, dterms2)
        assert np.linalg.norm(sensitivity_lib.evaluate_dketa_depsk(hess0, dterms3, eta0, eps0, deps)) > 0

        assert_array_almost_equal(
            np.einsum('ijkl,j,k,l', true_d3eta_deps3(eps0), deps, deps, deps),
            sensitivity_lib.evaluate_dketa_depsk(hess0, dterms3, eta0, eps0, deps))

        return True
        ###################################
        # Test the Taylor series itself.

        taylor_expansion = \
            ParametricSensitivityTaylorExpansion(
                objective_functor=model.get_objective,
                input_par=model.param,
                hyper_par=model.hyper_param,
                input_val0=eta0,
                hyper_val0=eps0,
                order=3,
                input_is_free=True,
                hyper_is_free=False,
                hess0=hess0)

        taylor_expansion.print_terms(k=3)

        d1 = np.einsum('ij,j', true_deta_deps(eps0), deps)
        d2 = np.einsum('ijk,j,k', true_d2eta_deps2(eps0), deps, deps)
        d3 = np.einsum('ijkl,j,k,l', true_d3eta_deps3(eps0), deps, deps, deps)

        assert_array_almost_equal(
            d1, taylor_expansion.evaluate_dkinput_dhyperk(deps, k=1))

        assert_array_almost_equal(
            d2, taylor_expansion.evaluate_dkinput_dhyperk(deps, k=2))

        assert_array_almost_equal(
            d3, taylor_expansion.evaluate_dkinput_dhyperk(deps, k=3))

        assert_array_almost_equal(
            eta0 + d1, taylor_expansion.evaluate_taylor_series(deps, max_order=1))

        assert_array_almost_equal(
            eta0 + d1 + 0.5 * d2,
            taylor_expansion.evaluate_taylor_series(deps, max_order=2))

        assert_array_almost_equal(
            eta0 + d1 + d2 / 2 + d3 / 6,
            taylor_expansion.evaluate_taylor_series(deps, max_order=3))

        assert_array_almost_equal(
            eta0 + d1 + d2 / 2 + d3 / 6,
            taylor_expansion.evaluate_taylor_series(deps))


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

import paragami

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from numpy.testing import assert_array_almost_equal
import unittest

def PytreesL1(p1, p2):
    diff_tree = jtu.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), p1, p2)
    return jtu.tree_reduce(lambda x, y: x + y, diff_tree, 0.0)



def GetSymMat(dim, rng_key):
    a = jax.random.uniform(rng_key, (dim, dim))
    a = 0.5 * (a + a.T)
    return a


def GetTestParameter():
    # Don't do flaky tests
    rng_key = jax.random.key(42)

    par = {}
    par['a'] = 5.0
    par['b'] = {}
    par['b']['b1'] = jnp.arange(3) + 1.0
    rng_key, rng_key2 = jax.random.split(rng_key)
    par['b']['b2'] = GetSymMat(3, rng_key2) + jnp.eye(3)

    return par


def GetTestBijector(par):
    par_bij = paragami.GetDefaultBijectors(par)
    par_bij['a'] = tfp.bijectors.Exp()
    par_bij['b']['b2'] = tfp.bijectors.CholeskyOuterProduct()
    return par_bij





class TestBijector(unittest.TestCase):
    def _CheckBijector(self, par, par_bij, tol=1e-8):
        par_free = paragami.Unconstrain(par, par_bij)
        par2 = paragami.Constrain(par_free, par_bij)
        err = PytreesL1(par, par2)
        self.assertTrue(err < tol)

    def test_constraints(self):
        par = GetTestParameter()
        par_bij = GetTestBijector(par)
        par_free = paragami.Unconstrain(par, par_bij)
        par2, ldj = paragami.Constrain(par_free, par_bij, compute_log_det_jac=True)
        self._CheckBijector(par, par_bij)

    def test_functions(self):
        par = GetTestParameter()
        par_bij = GetTestBijector(par)

        def Fun(par, x, y):
            return x * par['a'] + y * jnp.sum(par['b']['b1']) + \
                jnp.sum(jnp.linalg.eigvalsh(par['b']['b2']))

        FreeFun = paragami.UnconstrainObjective(Fun, par_bij)

        free_par = paragami.Unconstrain(par, par_bij)
        self.assertTrue(FreeFun(free_par, 2, y=3) == Fun(par, y=3, x=2))


if __name__ == '__main__':
    unittest.main()

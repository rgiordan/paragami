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


def TestBijector(par, par_bij, tol=1e-8):
    par_free = paragami.Unconstrain(par, par_bij)
    par2 = paragami.Constrain(par_free, par_bij)
    err = PytreesL1(par, par2)
    assert(err < tol)

def GetSymMat(dim, rng_key):
    a = jax.random.uniform(rng_key, (dim, dim))
    a = 0.5 * (a + a.T)
    return a



class TestParagami(unittest.TestCase):
    x = jnp.array([1.0])
    assert_array_almost_equal(x, x)


class TestBijector(unittest.TestCase):

    # Don't do flaky tests
    rng_key = jax.random.key(42)

    par = {}
    par['a'] = 5.0
    par['b'] = {}
    par['b']['b1'] = jnp.arange(3) + 1.0
    rng_key, rng_key2 = jax.random.split(rng_key)
    par['b']['b2'] = GetSymMat(3, rng_key2) + jnp.eye(3)


    par_bij = paragami.GetDefaultBijectors(par)
    par_bij['a'] = tfp.bijectors.Exp()
    par_bij['b']['b2'] = tfp.bijectors.CholeskyOuterProduct()

    par_free = paragami.Unconstrain(par, par_bij)
    par2, ldj = paragami.Constrain(par_free, par_bij, compute_log_det_jac=True)
    TestBijector(par, par_bij)


if __name__ == '__main__':
    unittest.main()

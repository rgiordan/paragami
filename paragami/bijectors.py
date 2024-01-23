
import gzip
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import json
from tensorflow_probability.substrates import jax as tfp


def GetDefaultBijectors(par):
    return jtu.tree_map(lambda x: tfp.bijectors.Identity(), par)


def Unconstrain(par, par_bij):
    par_free = jtu.tree_map(lambda x, bij: bij.inverse(x), par, par_bij)
    return par_free


def Constrain(par_free, par_bij, compute_log_det_jac=False):
    par = jtu.tree_map(lambda x, bij: bij.forward(x), par_free, par_bij)
    if compute_log_det_jac:
        ldj_tree = jtu.tree_map(lambda x, bij: bij.inverse_log_det_jacobian(x), par, par_bij)
        ldj = jtu.tree_reduce(lambda x, y: x + jnp.sum(y), ldj_tree, 0.0)
        return par, ldj
    else:
        return par


def PytreesL1(p1, p2):
    diff_tree = jtu.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), p1, p2)
    return jtu.tree_reduce(lambda x, y: x + y, diff_tree, 0.0)


def TestBijector(par, par_bij, tol=1e-8):
    par_free = Unconstrain(par, par_bij)
    par2 = Constrain(par_free, par_bij)
    err = PytreesL1(par, par2)
    assert(err < tol)


def UnconstrainObjective(Fun, par_bij, include_log_det_jac=False):
    def FreeFun(par_free, *vargs, **kargs):
        par = Constrain(par_free, par_bij, compute_log_det_jac=include_log_det_jac)
        if include_log_det_jac:
            ldj = par[1]
            par = par[0]
        fval = Fun(par, *vargs, **kargs)
        if include_log_det_jac:
            return fval + ldj
        else:
            return fval
    return FreeFun

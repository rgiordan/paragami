import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive, defvjp, defjvp

from autograd.numpy.linalg import slogdet, solve, inv
from functools import partial

# Most of these are copied with minimal modification from
# https://github.com/HIPS/autograd/blob/65c21e2/autograd/numpy/linalg.py

# transpose by swapping last two dimensions
def T(x): return np.swapaxes(x, -1, -2)

def inv_jvp(g, ans, x):
    dot = np.dot if ans.ndim == 2 else partial(np.einsum, '...ij,...jk->...ik')
    return -dot(dot(ans, g), ans)

defjvp(inv, inv_jvp)

# The JVP is the Jacobian times the input.
# The gradient g should be the same shape as the input.
# The output should be the same shape as the output.
def jvp_solve(argnum, g, ans, a, b):
    print('\njvp_solve:\n')
    print('a', a.shape)
    print('b', b.shape)
    print('g', g.shape)
    print('ans', ans.shape)
    updim = lambda x: x if x.ndim == a.ndim else x[...,None]
    dot = np.dot if a.ndim == 2 else partial(np.einsum, '...ij,...jk->...ik')
    if argnum == 0:
        print('an=0')
        return -dot(np.linalg.solve(a, g), ans)
    else:
        print('an=1')
        return np.linalg.solve(a, g)

defjvp(solve, partial(jvp_solve, 0), partial(jvp_solve, 1))

# def solve_jvp_argnum0(g, ans, x, y):
#     return -1 * np.linalg.solve(x, g) @ ans
#
# def solve_jvp_argnum1(g, ans, x, y):
#     return np.linalg.solve(x, g)
#
# defjvp(solve, solve_jvp_argnum0, solve_jvp_argnum1)

def slogdet_jvp(g, ans, x):
    # Due to https://github.com/HIPS/autograd/issues/115
    # and https://github.com/HIPS/autograd/blob/65c21e/tests/test_numpy.py#L302
    # it does not seem easy to take the trace of the last two dimensions of
    # a multi-dimensional array at this time.
    if len(x.shape) > 2:
        raise ValueError('JVP is only supported for 2d input.')
    return 0, np.trace(np.linalg.solve(x.T, g.T))

defjvp(slogdet, slogdet_jvp)

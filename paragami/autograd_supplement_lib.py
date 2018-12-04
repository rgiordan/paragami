import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive, defvjp, defjvp

from autograd.numpy.linalg import slogdet, solve, inv

# TODO: Handle broadcasting like the core autograd does.

def inv_jvp(g, ans, x):
    return -1 * ans @ g @ ans

defjvp(inv, inv_jvp)

def solve_jvp_argnum0(g, ans, x, y):
    return -1 * np.linalg.solve(x, g) @ ans

def solve_jvp_argnum1(g, ans, x, y):
    return np.linalg.solve(x, g)

defjvp(solve, solve_jvp_argnum0, solve_jvp_argnum1)

def slogdet_jvp(g, ans, x):
    # Due to https://github.com/HIPS/autograd/issues/115
    # and https://github.com/HIPS/autograd/blob/65c21e/tests/test_numpy.py#L302
    # it does not seem easy to take the trace of the last two dimensions of
    # a multi-dimensional array at this time.
    if len(x.shape) > 2:
        raise ValueError('JVP is only supported for 2d input.')
    return 0, np.trace(np.linalg.solve(x.T, g.T))

defjvp(slogdet, slogdet_jvp)

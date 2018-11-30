import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive, defvjp, defjvp

from autograd.numpy.linalg import slogdet, solve, inv

# TODO: Handle broadcasting like the core autograd does.

def inv_jvp(g, ans, x):
    return -1 * ans @ g @ ans

defjvp(inv, inv_jvp)

def solve_jvp_0(g, ans, x, y):
    return -1 * np.linalg.solve(x, g) @ ans

def solve_jvp_1(g, ans, x, y):
    return np.linalg.solve(x, g)

defjvp(solve, solve_jvp_0, solve_jvp_1)

def slogdet_jvp(g, ans, x):
    return 0, np.trace(np.linalg.solve(x.T, g.T))

defjvp(slogdet, slogdet_jvp)

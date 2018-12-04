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

def jvp_solve(argnum, g, ans, a, b):
    print('\njvp_solve:\n')
    print('a', a.shape)
    print('b', b.shape)
    print('g', g.shape)
    print('ans', ans.shape)
    def broadcast_matmul(a, b):
        return \
            np.matmul(a, b) if b.ndim == a.ndim \
            else np.matmul(a, b[..., None])[..., 0]
    if argnum == 0:
        print('an=0')
        foo = np.linalg.solve(a, g)
        print('np.linalg.solve(a, g)', foo.shape)
        # return -dot(np.linalg.solve(a, g), updim(ans))
        return -broadcast_matmul(np.linalg.solve(a, g), ans)
    else:
        print('an=1')
        return np.linalg.solve(a, g)

defjvp(solve, partial(jvp_solve, 0), partial(jvp_solve, 1))


def slogdet_jvp(g, ans, x):
    # Due to https://github.com/HIPS/autograd/issues/115
    # and https://github.com/HIPS/autograd/blob/65c21e/tests/test_numpy.py#L302
    # it does not seem easy to take the trace of the last two dimensions of
    # a multi-dimensional array at this time.
    if len(x.shape) > 2:
        raise ValueError('JVP is only supported for 2d input.')
    return 0, np.trace(np.linalg.solve(x.T, g.T))

defjvp(slogdet, slogdet_jvp)

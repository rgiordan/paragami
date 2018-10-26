
from paragami.base_patterns import Pattern

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import copy
import math

from autograd.core import primitive, defvjp, defjvp

import scipy as osp
from scipy.sparse import coo_matrix

# The first index is assumed to index simplicial observations.
def _constrain_simplex_matrix(free_mat):
    # The first column is the reference value.
    free_mat_aug = np.hstack(
        [np.full((free_mat.shape[0], 1), 0.), free_mat])
    # Note that autograd needs to update their logsumexp to be in special
    # not misc before this can be changed.
    log_norm = np.expand_dims(sp.misc.logsumexp(free_mat_aug, 1), axis=1)
    return np.exp(free_mat_aug - log_norm)


def _unconstrain_simplex_matrix(simplex_mat):
    return np.log(simplex_mat[:, 1:]) - \
           np.expand_dims(np.log(simplex_mat[:, 0]), axis=1)


def _constrain_simplex_vector(free_vec):
    return _constrain_simplex_matrix(np.expand_dims(free_vec, 0)).flatten()

# TODO: some compuatation is shared between the Jacobian and Hessian.

# # The Jacobian of the constraint is most easily calculated as a function
# # of the constrained moments.
# def constrain_grad_from_moment(z):
#     z_last = z[1:]
#     z_jac = -1 * np.outer(z, z_last)
#     for k in range(1, len(z)):
#         z_jac[k, k - 1] += z[k]
#     return z_jac
#
# # The Hessian of the constraint is most easily calculated as a function
# # of the constrained moments.
# def constrain_hess_from_moment(z):
#     # See the notes simplex_derivatives.lyx for a derivation.
#     z_last = z[1:]
#     z_outer = np.expand_dims(2 * np.outer(z_last, z_last), axis=0)
#     z_hess = np.tile(z_outer, (len(z), 1, 1))
#     # The first index is different.
#     for k in range(1, len(z)):
#         z_hess[0, k - 1, k - 1] += -z[k]
#
#     z_hess[0] *= z[0]
#
#     for k in range(1, len(z)):
#         z_hess[k, k - 1, :] += -z_last
#         z_hess[k, :, k - 1] += -z_last
#         for j in range(1, len(z)):
#             if j == k:
#                 z_hess[k, j - 1, j - 1] += 1 - z[j]
#             else:
#                 z_hess[k, j - 1, j - 1] += -z[j]
#         z_hess[k, :, :] *= z[k]
#
#     return z_hess


# This is actually a vector of simplexes.  The first index of the shape
# is which simplex, and the second index is the element within the simplex.
# TODO: make this more general.
class SimplexArrayPattern(Pattern):
    def __init__(self, simplex_size, array_shape, validate=True):
        self.__simplex_size = int(simplex_size)
        if self.__simplex_size <= 1:
            raise ValueError('simplex_size must be >= 2.')
        self.__array_shape = array_shape
        self.__shape = self.__array_shape + (self.__simplex_size, )
        self.__free_shape = self.__array_shape + (self.__simplex_size - 1, )
        self.validate = validate
        super().__init__(np.prod(self.__shape), np.prod(self.__free_shape))

    def __str__(self):
        return 'SimplexArrayPattern {} of {}-d simplices'.format(
            self.__array_shape, self.__simplex_size)

    def array_shape(self):
        return self.__array_shape

    def simplex_size(self):
        return self.__simplex_size

    def shape(self):
        return self.__shape

    def __eq__(self, other):
        return \
            (type(self) == type(other)) & \
            (self.array_shape() == other.array_shape()) & \
            (self.simplex_size() == other.simplex_size())

    def empty(self, valid):
        if valid:
            return np.full(self.__shape, 1.0 / self.simplex_size)
        else:
            return np.empty(self.__shape)

    def validate_folded(self, folded_val):
        if folded_val.shape != self.__shape:
            return False
        if self.validate:
            if np.any(folded_val < 0):
                return False
            simplex_sums = np.sum(folded_val, axis=-1)
            if np.any(np.abs(simplex_sums - 1) > 1e-12):
                return False
        return True

    def fold(self, flat_val, free):
        flat_size = self.flat_size(free)
        if len(flat_val) != flat_size:
            raise ValueError('flat_val is the wrong length.')
        if free:
            free_mat = np.reshape(free_val, self.__free_shape)
            return _constrain_simplex_matrix(free_mat)
        else:
            folded_val = np.reshape(vec_val, self.__shape)
            self.validate_folded(folded_val)
            return folded_val

    def flatten(self, folded_val, free):
        self.validate_folded(folded_val)
        if free:
            return _unconstrain_simplex_matrix(folded_val).flatten()
        else:
            return folded_val.flatten()

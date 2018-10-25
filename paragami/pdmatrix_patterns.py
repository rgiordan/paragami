from base_patterns import Pattern

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import copy
import math

from autograd.core import primitive, defvjp, defjvp

#import itertools

#import scipy as osp
#from scipy.sparse import coo_matrix

# Return the linear index of an element of a symmetric matrix
# where the triangular part has been stacked into a vector.
#
# Uses 0-indexing. (row, col) = (k1, k2)
def _sym_index(k1, k2):
    def ld_ind(k1, k2):
        return int(k2 + k1 * (k1 + 1) / 2)

    if k2 <= k1:
        return ld_ind(k1, k2)
    else:
        return ld_ind(k2, k1)


# Map a matrix
#
# [ x11 x12 ... x1n ]
# [ x21 x22     x2n ]
# [...              ]
# [ xn1 ...     xnn ]
#
# to the vector
#
# [ x11, x21, x22, x31, ..., xnn ].
#
# The entries above the diagonal are ignored.
def _vectorize_ld_matrix(mat):
    nrow, ncol = np.shape(mat)
    if nrow != ncol: raise ValueError('mat must be square')
    return mat[np.tril_indices(nrow)]


# Map a vector
#
# [ v1, v2, ..., vn ]
#
# to the symmetric matrix
#
# [ v1 ...          ]
# [ v2 v3 ...       ]
# [ v4 v5 v6 ...    ]
# [ ...             ]
#
# where the values above the diagonal are determined by symmetry.
#
# Because we cannot use autograd with array assignment, just define the
# vector jacobian product directly.
@primitive
def _unvectorize_ld_matrix(vec):
    mat_size = int(0.5 * (math.sqrt(1 + 8 * vec.size) - 1))
    if mat_size * (mat_size + 1) / 2 != vec.size: \
        raise ValueError('Vector is an impossible size')
    mat = np.zeros((mat_size, mat_size))
    for k1 in range(mat_size):
        for k2 in range(k1 + 1):
            mat[k1, k2] = vec[_sym_index(k1, k2)]
    return mat

def _unvectorize_ld_matrix_vjp(g):
    assert g.shape[0] == g.shape[1]
    return vectorize_ld_matrix(g)

defvjp(_unvectorize_ld_matrix,
       lambda ans, vec: lambda g: _unvectorize_ld_matrix_vjp(g))

def _unvectorize_ld_matrix_jvp(g):
    return _unvectorize_ld_matrix(g)

defjvp(_unvectorize_ld_matrix, lambda g, ans, x : _unvectorize_ld_matrix_jvp(g))


def _exp_matrix_diagonal(mat):
    assert mat.shape[0] == mat.shape[1]
    # make_diagonal() is only defined in the autograd version of numpy
    mat_exp_diag = np.make_diagonal(
        np.exp(np.diag(mat)), offset=0, axis1=-1, axis2=-2)
    mat_diag = np.make_diagonal(np.diag(mat), offset=0, axis1=-1, axis2=-2)
    return mat_exp_diag + mat - mat_diag


def _log_matrix_diagonal(mat):
    assert mat.shape[0] == mat.shape[1]
    # make_diagonal() is only defined in the autograd version of numpy
    mat_log_diag = np.make_diagonal(
        np.log(np.diag(mat)), offset=0, axis1=-1, axis2=-2)
    mat_diag = np.make_diagonal(np.diag(mat), offset=0, axis1=-1, axis2=-2)
    return mat_log_diag + mat - mat_diag


def _pack_posdef_matrix(mat, diag_lb=0.0):
    k = mat.shape[0]
    mat_lb = mat - np.make_diagonal(
        np.full(k, diag_lb), offset=0, axis1=-1, axis2=-2)
    return _vectorize_ld_matrix(
        _log_matrix_diagonal(np.linalg.cholesky(mat_lb)))


def _unpack_posdef_matrix(free_vec, diag_lb=0.0):
    mat_chol = _exp_matrix_diagonal(_unvectorize_ld_matrix(free_vec))
    mat = np.matmul(mat_chol, mat_chol.T)
    k = mat.shape[0]
    return mat + np.make_diagonal(
        np.full(k, diag_lb), offset=0, axis1=-1, axis2=-2)


# Convert a vector containing the lower diagonal portion of a symmetric
# matrix into the full symmetric matrix.
def _unvectorize_symmetric_matrix(vec_val):
    ld_mat = _unvectorize_ld_matrix(vec_val)
    mat_val = ld_mat + ld_mat.transpose()
    # We have double counted the diagonal.  For some reason the autograd
    # diagonal functions require axis1=-1 and axis2=-2
    mat_val = mat_val - \
        np.make_diagonal(np.diagonal(ld_mat, axis1=-1, axis2=-2),
                         axis1=-1, axis2=-2)
    return mat_val


# def _pos_def_matrix_free_to_vector(free_val, diag_lb=0.0):
#     mat_val = _unpack_posdef_matrix(free_val, diag_lb=diag_lb)
#     return _vectorize_ld_matrix(mat_val)
#
# pos_def_matrix_free_to_vector_jac = \
#     autograd.jacobian(_pos_def_matrix_free_to_vector)
# pos_def_matrix_free_to_vector_hess = \
#     autograd.hessian(_pos_def_matrix_free_to_vector)

class PDMatrixPattern(Pattern):
    def __init__(self, size, diag_lb=0.0, validate=True):
        self.__size = int(size)
        self.__diag_lb = diag_lb
        self.validate = validate
        if diag_lb < 0:
            raise ValueError('The diagonal lower bound diag_lb must be >-= 0.')

        vec_size = int(size * (size + 1) / 2)
        super().__init__(vec_size, vec_size)

    def __str__(self):
        return 'PDMatrix {}x{} (diag_lb = {})'.format(
            self.__size, self.__size, self.__diag_lb)

    def __eq__(self, other):
        return \
            (self.size() == other.size()) & \
            (self.diag_lb() == other.diag_lb())

    def size(self):
        return self.__size

    def diag_lb(self):
        return self.__diag_lb

    def empty(self, valid):
        if valid:
            return np.eye(self.__size) * (self.__diag_lb + 1)
        else:
            return np.empty((self.__size, self.__size))

    def validate_folded(self, folded_val):
        if folded_val.shape != (self.__size, self.__size):
            raise ValueError('Wrong shape for PDMatrix.')
        if self.validate:
            if np.any(np.diag(folded_val) < self.__diag_lb):
                raise ValueError('Diagonal is less than the lower bound.')
            if not (folded_val.transpose() == folded_val).all():
                raise ValueError('Matrix is not symmetric')
            try:
                chol = np.linalg.cholesky(folded_val)
            except LinAlgError:
                raise ValueError('Matrix is not positive definite.')

    def _free_fold(self, free_flat_val):
        if free_flat_val.size != self._free_flat_length:
            # TODO: make these errors consistently worded.
            raise ValueError('Wrong length for PDMatrix free flat value.')
        return _unpack_posdef_matrix(free_flat_val, diag_lb=self.__diag_lb)

    def _free_flatten(self, folded_val):
        self.validate_folded(folded_val)
        return _pack_posdef_matrix(folded_val, diag_lb=self.__diag_lb)

    def _notfree_fold(self, flat_val):
        if flat_val.size != self._flat_length:
            raise ValueError('Wrong length for PDMatrix flat value.')
        folded_val = _unvectorize_symmetric_matrix(flat_val)
        self.validate_folded(folded_val)
        return folded_val

    def _notfree_flatten(self, folded_val):
        self.validate_folded(folded_val)
        return _vectorize_ld_matrix(folded_val)

    def flatten(self, folded_val, free):
        if free:
            return self._free_flatten(folded_val)
        else:
            return self._notfree_flatten(folded_val)

    def fold(self, flat_val, free):
        flat_val = np.atleast_1d(flat_val)
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        if free:
            return self._free_fold(flat_val)
        else:
            return self._notfree_fold(flat_val)

#
# # In keeping with the vector version, the last two indices are indices
# # of the matrix.
# class PosDefMatrixParamArray(object):
#     def __init__(self, name='', array_shape=(1), matrix_size=2, diag_lb=0.0, val=None):
#         self.name = name
#         self.__matrix_size = int(matrix_size)
#         self.__matrix_shape = np.array([ int(matrix_size), int(matrix_size) ])
#         self.__array_shape = array_shape
#         self.__array_ranges = [ range(0, t) for t in self.__array_shape ]
#         self.__array_length = np.prod(self.__array_shape)
#         self.__shape = np.append(self.__array_shape, self.__matrix_shape)
#         # __vec_size is the size of a single matrix in vector form.
#         self.__vec_size = int(matrix_size * (matrix_size + 1) / 2)
#         self.__diag_lb = diag_lb
#         assert diag_lb >= 0
#         if val is None:
#             default_val = np.diag(np.full(self.__matrix_size, diag_lb + 1.0))
#             self.__val = np.broadcast_to(default_val, self.__shape)
#         else:
#             self.set(val)
#
#     def __str__(self):
#         return self.name + ':\n' + str(self.__val)
#     def names(self):
#         return [ self.name ]
#     def dictval(self):
#         return self.__val.tolist()
#
#     def set(self, val):
#         if (val.shape != self.__shape).all():
#             print(val.shape)
#             print(self.__shape)
#             raise ValueError('Array is the wrong size')
#         self.__val = val
#
#     def get(self):
#         return self.__val
#
#     # Get the elements in a free vector correpsonding to element obs of the
#     # array, where obs is a tuple indexing into the array of shape
#     # self.__array_shape.
#     def stacked_obs_slice(self, obs):
#         assert len(obs) == len(self.__array_shape)
#         linear_obs = np.ravel_multi_index(obs, self.__array_shape) * \
#                      self.__vec_size
#         return slice(linear_obs, linear_obs + self.__vec_size)
#
#     def set_free(self, free_val):
#         if free_val.size != self.free_size():
#             raise ValueError('Free value is the wrong length')
#         new_val = \
#             np.reshape(np.array([ _unpack_posdef_matrix(
#                 free_val[self.stacked_obs_slice(obs)], diag_lb=self.__diag_lb) \
#               for obs in itertools.product(*self.__array_ranges) ]),
#               self.__shape)
#         self.__val = new_val
#
#     # Return an array of the function mat_func applied to each matrix in
#     # the array.
#     def apply_matrix_function(self, mat_func):
#         mat_func_array = np.array([
#                 mat_func(self.__val[obs]) \
#                     for obs in itertools.product(*self.__array_ranges) ])
#
#         # Assume that each output is the same shape.
#         new_shape = self.__array_shape + mat_func_array[0].shape
#         return np.reshape(mat_func_array, new_shape)
#
#     def get_free(self):
#         # I don't know why this doesn't work, but it gives an autograd error
#         # about assigning in arrays:
#         #return np.hstack(self.apply_matrix_function(pack_posdef_matrix))
#         return np.hstack(np.array([ \
#             _pack_posdef_matrix(self.__val[obs], diag_lb=self.__diag_lb) \
#                     for obs in itertools.product(*self.__array_ranges)]))
#
#     def set_vector(self, vec_val):
#         if len(vec_val) != self.vector_size():
#             raise ValueError('Vector value is the wrong length')
#         self.__val = \
#             np.reshape(np.array([ _unvectorize_symmetric_matrix(
#                     vec_val[self.stacked_obs_slice(obs)]) \
#                 for obs in itertools.product(*self.__array_ranges) ]),
#               self.__shape)
#
#     def get_vector(self):
#         vec_val = np.hstack([ _vectorize_ld_matrix(
#             self.__val[obs]) \
#             for obs in itertools.product(*self.__array_ranges) ])
#         #print('get vector ', vec_val)
#         return vec_val
#
#     def free_to_vector(self, free_val):
#         self.set_free(free_val)
#         return self.get_vector()
#
#     def free_to_vector_jac(self, free_val):
#         jac_rows = []
#         jac_cols = []
#         grads = []
#         vec_rows = range(self.__vec_size)
#         # This is the shape of an array of the same size as self.__array_size
#         # of packed vectors.  We'll use it to pick out
#         packed_shape = self.__array_shape + (self.__vec_size,)
#
#         for obs in itertools.product(*self.__array_ranges):
#             # This seems like a convoluted expression, but it is what
#             # is required by ravel_multi_index.
#             array_inds = tuple([ [t] for t in obs]) + (vec_rows,)
#             vec_inds = np.ravel_multi_index(array_inds, packed_shape)
#             row_jac = pos_def_matrix_free_to_vector_jac(free_val[vec_inds])
#             for vec_row in vec_rows:
#                 for free_row in vec_rows:
#                     jac_rows.append(vec_inds[vec_row])
#                     jac_cols.append(vec_inds[free_row])
#                     grads.append(row_jac[vec_row, free_row])
#
#         return coo_matrix(
#             (grads, (jac_rows, jac_cols)),
#             (self.vector_size(), self.free_size()))
#
#     def free_to_vector_hess(self, free_val):
#         hessians = []
#         vec_rows = range(self.__vec_size)
#         packed_shape = self.__array_shape + (self.__vec_size,)
#         hess_shape = (self.__array_length * self.__vec_size,
#                       self.__array_length * self.__vec_size)
#         for obs in itertools.product(*self.__array_ranges):
#         #for row in range(self.__length):
#             #vec_inds = np.ravel_multi_index([[row], vec_rows], packed_shape)
#             # vec_inds = self.stacked_obs_slice(obs)
#             array_inds = tuple([ [t] for t in obs]) + (vec_rows,)
#             vec_inds = np.ravel_multi_index(array_inds, packed_shape)
#             row_hess = pos_def_matrix_free_to_vector_hess(free_val[vec_inds])
#             for vec_row in vec_rows:
#                 hess_rows = []
#                 hess_cols = []
#                 hess_vals = []
#                 for free_row1 in vec_rows:
#                     for free_row2 in vec_rows:
#                         hess_rows.append(vec_inds[free_row1])
#                         hess_cols.append(vec_inds[free_row2])
#                         hess_vals.append(row_hess[vec_row, free_row1, free_row2])
#
#                 # It is important that this traverse vec_inds in order because
#                 # we simply append the hessians to the end.
#                 hessians.append(
#                     coo_matrix((hess_vals, (hess_rows, hess_cols)), hess_shape))
#
#         return hessians
#
#     def length(self):
#         return self.__length
#     def matrix_size(self):
#         return self.__matrix_size
#     def free_size(self):
#         return self.__vec_size * self.__array_length
#     def vector_size(self):
#         return self.__vec_size * self.__array_length
#     def get_array_ranges(self):
#         return self.__array_ranges

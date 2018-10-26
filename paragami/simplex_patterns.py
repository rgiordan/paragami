
from paragami.base_patterns import Pattern

import autograd
import autograd.numpy as np
import autograd.scipy as sp

def _constrain_simplex_matrix(free_mat):
    # The first column is the reference value.  Append a column of zeros
    # to each simplex representing this reference value.
    reference_col = np.expand_dims(np.full(free_mat.shape[0:-1], 0), axis=-1)
    free_mat_aug = np.concatenate([reference_col, free_mat], axis=-1)

    # Note that autograd needs to update their logsumexp to be in special
    # not misc before this can be changed.
    log_norm = np.expand_dims(sp.misc.logsumexp(free_mat_aug, axis=-1), axis=-1)
    return np.exp(free_mat_aug - log_norm)


def _unconstrain_simplex_matrix(simplex_mat):
    return np.log(simplex_mat[..., 1:]) - \
           np.expand_dims(np.log(simplex_mat[..., 0]), axis=-1)


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
            return np.full(self.__shape, 1.0 / self.__simplex_size)
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
        flat_size = self.flat_length(free)
        if len(flat_val) != flat_size:
            raise ValueError('flat_val is the wrong length.')
        if free:
            free_mat = np.reshape(flat_val, self.__free_shape)
            return _constrain_simplex_matrix(free_mat)
        else:
            folded_val = np.reshape(flat_val, self.__shape)
            self.validate_folded(folded_val)
            return folded_val

    def flatten(self, folded_val, free):
        self.validate_folded(folded_val)
        if free:
            return _unconstrain_simplex_matrix(folded_val).flatten()
        else:
            return folded_val.flatten()

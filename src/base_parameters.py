# This is a draft of the new API.

import math
import copy
import numbers

import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive

from collections import OrderedDict

import scipy as osp
from scipy.sparse import coo_matrix, csr_matrix, block_diag

import warnings


class Pattern(object):
    def __init__(self, name, flat_length, free_flat_length):
        self.name = name
        self._flat_length = flat_length
        self._free_flat_length = free_flat_length

    def __str__(self):
        return self.name

    # Maybe this should be to / from JSON.
    def serialize(self):
        raise NotImplementedError()

    def unserialize(self):
        raise NotImplementedError()

    def fold(self, flat_val, free):
        raise NotImplementedError()

    def flatten(self, folded_val, free):
        raise NotImplementedError()

    def flat_length(self, free):
        if free:
            return self._free_flat_length
        else:
            return self._flat_length

    def empty(self, valid):
        raise NotImplementedError()

    def random(self):
        return self.fold(np.random.random(self._free_flat_length), free=True)


##################################
# Array parameter

def _unconstrain_array(array, lb, ub):
    if not (array <= ub).all():
        raise ValueError('Elements larger than the upper bound')
    if not (array >= lb).all():
        raise ValueError('Elements smaller than the lower bound')
    if ub <= lb:
        raise ValueError('Upper bound must be greater than lower bound')
    if ub == float("inf"):
        if lb == -float("inf"):
            # For consistent behavior, never return a reference.
            return copy.deepcopy(array)
        else:
            return np.log(array - lb)
    else: # the upper bound is finite
        if lb == -float("inf"):
            return -1 * np.log(ub - array)
        else:
            return np.log(array - lb) - np.log(ub - array)


def _constrain_array(free_array, lb, ub):
    if ub <= lb:
        raise ValueError('Upper bound must be greater than lower bound')
    if ub == float("inf"):
        if lb == -float("inf"):
            # For consistency, never return a reference.
            return copy.deepcopy(free_array)
        else:
            return np.exp(free_array) + lb
    else: # the upper bound is finite
        if lb == -float("inf"):
            return ub - np.exp(-1 * free_array)
        else:
            exp_vec = np.exp(free_array)
            return (ub - lb) * exp_vec / (1 + exp_vec) + lb


def _get_inbounds_value(lb, ub):
    assert lb < ub
    if lb > -float('inf') and ub < float('inf'):
        return 0.5 * (ub - lb)
    else:
        if lb > -float('inf'):
            # The upper bound is infinite.
            return lb + 1.0
        elif ub < float('inf'):
            # The lower bound is infinite.
            return ub - 1.0
        else:
            # Both are infinite.
            return 0.0


class ArrayPattern(Pattern):
    def __init__(
        self, name='', shape=(1, ),
        lb=-float("inf"), ub=float("inf"), bound_checking=True):

        self.bound_checking = bound_checking
        self.__shape = shape
        self.__lb = lb
        self.__ub = ub
        assert lb >= -float('inf')
        assert ub <= float('inf')
        if lb >= ub:
            raise ValueError(
                'Upper bound ub must strictly exceed lower bound lb')

        free_flat_length = flat_length = int(np.product(self.__shape))

        super().__init__(name, flat_length, free_flat_length)

    def serialize(self):
        return self.__val.tolist()

    def empty(self, valid):
        if valid:
            return np.full(self.__shape, _get_inbounds_value(self.__lb, self.__ub))
        else:
            return np.empty(self.__shape)

    def validate_folded(self, folded_val):
        folded_val = np.atleast_1d(folded_val)
        if folded_val.shape != self.shape():
            raise ValueError('Wrong size for array ' + self.name + \
                             ' Expected shape: ' + str(self.shape()) + \
                             ' Got shape: ' + str(folded_val.shape))
        if self.bound_checking:
            if (np.array(folded_val < self.__lb)).any():
                raise ValueError('Value beneath lower bound.')
            if (np.array(folded_val > self.__ub)).any():
                raise ValueError('Value above upper bound.')

    def _free_fold(self, free_flat_val):
        if free_flat_val.size != self._free_flat_length:
            error_string = \
                'Wrong size for array {}.  Expected {}, got {}'.format(
                    self.name,
                    str(self._free_flat_length),
                    str(free_flat_val.size))
            raise ValueError(error_string)
        constrained_array = _constrain_array(free_flat_val, self.__lb, self.__ub)
        return constrained_array.reshape(self.__shape)

    def _free_flatten(self, folded_val):
        self.validate_folded(folded_val)
        return _unconstrain_array(folded_val, self.__lb, self.__ub).flatten()

    def _notfree_fold(self, flat_val):
        if flat_val.size != self._flat_length:
            error_string = \
                'Wrong size for array {}.  Expected {}, got {}'.format(
                    self.name, str(self._flat_length), str(flat_val.size))
            raise ValueError(error_string)
        folded_val = flat_val.reshape(self.__shape)
        self.validate_folded(folded_val)
        return folded_val

    def _notfree_flatten(self, folded_val):
        self.validate_folded(folded_val)
        return folded_val.flatten()

    def fold(self, flat_val, free):
        flat_val = np.atleast_1d(flat_val)
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        if free:
            return self._free_fold(flat_val)
        else:
            return self._notfree_fold(flat_val)

    def flatten(self, folded_val, free):
        if free:
            return self._free_flatten(folded_val)
        else:
            return self._notfree_flatten(folded_val)

    def shape(self):
        return self.__shape

    def flat_length(self, free):
        if free:
            return self._free_flat_length
        else:
            return self._flat_length


##########################
# Dictionary pattern.

class OrderedDictPattern(Pattern):
    def __init__(self, name):
        self.__pattern_dict = OrderedDict()
        super().__init__(name, 0, 0)

    def __str__(self):
        return self.name + ':\n' + \
            '\n'.join([ '\t' + str(pattern) \
                for pattern in self.__pattern_dict.values() ])

    def __getitem__(self, key):
        return self.__pattern_dict[key]

    def __setitem__(self, pattern_name, pattern):
        self.__pattern_dict[pattern_name] = pattern
        self._flat_length += pattern.flat_length(free=False)
        self._free_flat_length += pattern.flat_length(free=True)

    def __delitem__(self, pattern_name):
        pattern = self.__pattern_dict[pattern_name]
        self._flat_length -= pattern.flat_length(free=False)
        self._free_flat_length -= pattern.flat_length(free=True)
        self.__pattern_dict.pop(pattern_name)

    def empty(self, valid):
        empty_val = OrderedDict()
        for pattern_name, pattern in self.__pattern_dict.items():
            empty_val[pattern_name] = pattern.empty(valid)
        return empty_val

    def serialize(self):
        result = {}
        for pattern in self.__pattern_dict.values():
            result[pattern.name] = pattern.serialize()
        return result

    def fold(self, flat_val, free):
        flat_val = np.atleast_1d(flat_val)
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        flat_length = self.flat_length(free)
        if flat_val.size != flat_length:
            error_string = \
                'Wrong size for parameter {}.  Expected {}, got {}'.format(
                    self.name, str(flat_length), str(flat_val.size))
            raise ValueError(error_string)

        # TODO: add an option to do this -- and other operations -- in place.
        folded_val = OrderedDict()
        offset = 0
        for pattern_name, pattern in self.__pattern_dict.items():
            pattern_flat_length = pattern.flat_length(free)
            pattern_flat_val = flat_val[offset:(offset + pattern_flat_length)]
            offset += pattern_flat_length
            folded_val[pattern_name] = pattern.fold(pattern_flat_val, free)
        return folded_val

    def flatten(self, folded_val, free):
        flat_length = self.flat_length(free)
        offset = 0
        flat_val = np.full(flat_length, float('nan'))
        for pattern_name, pattern in self.__pattern_dict.items():
            pattern_flat_length = pattern.flat_length(free)
            flat_val[offset:(offset + pattern_flat_length)] = \
                pattern.flatten(folded_val[pattern_name], free)
            offset += pattern_flat_length
        return flat_val

    def flat_length(self, free):
        return self._free_flat_length if free else self._flat_length

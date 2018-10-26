# This is a draft of the new API.

import math
import copy
import numbers

import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive

from collections import OrderedDict
import itertools

import scipy as osp
from scipy.sparse import coo_matrix, csr_matrix, block_diag

import warnings


class Pattern(object):
    def __init__(self, flat_length, free_flat_length):
        self._flat_length = flat_length
        self._free_flat_length = free_flat_length

    def __str__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    # Fold: convert from a vector to the parameter shape.
    def fold(self, flat_val, free):
        raise NotImplementedError()

    # Flatten: convert from the parameter shape to a vector.
    def flatten(self, folded_val, free):
        raise NotImplementedError()

    # Get the size of the flattened version.
    def flat_length(self, free):
        if free:
            return self._free_flat_length
        else:
            return self._flat_length

    # Methods to generate valid values.
    def empty(self, valid):
        raise NotImplementedError()

    def random(self):
        return self.fold(np.random.random(self._free_flat_length), free=True)

    # These are currently not implemented, but we should.
    # Maybe this should be to / from JSON.
    def serialize(self):
        raise NotImplementedError()

    def unserialize(self, serialized_val):
        raise NotImplementedError()


##########################
# Dictionary of patterns.

class PatternDict(Pattern):
    def __init__(self):
        self.__pattern_dict = OrderedDict()
        super().__init__(0, 0)

    def __str__(self):
        pattern_strings = [
            '\t[' + key + '] = ' + str(self.__pattern_dict[key]) \
            for key in self.__pattern_dict ]
        return \
            'OrderedDict:\n' + \
            '\n'.join(pattern_strings)

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if self.__pattern_dict.keys() != other.keys():
            return False
        for pattern_name in self.__pattern_dict.keys():
            if self.__pattern_dict[pattern_name] != \
                other[pattern_name]:
                return False
        return True

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

    def keys(self):
        return self.__pattern_dict.keys()

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


##########################
# An array of a pattern.

class PatternArray(Pattern):
    def __init__(self, shape, base_pattern):
        self.__shape = shape
        self.__array_ranges = [ range(0, t) for t in self.__shape ]

        num_elements = np.prod(self.__shape)
        self.__base_pattern = base_pattern

        # Check whether the base_pattern takes values that are numpy arrays.
        # If they are, then the unfolded value will be a single numpy array
        # of shape __shape + base_pattern.empty().shape.
        # Otherwise, it will be an object array of shape __shape.
        empty_pattern = self.__base_pattern.empty(valid=False)
        if type(empty_pattern) is np.ndarray:
            self.__folded_pattern_shape = empty_pattern.shape
        else:
            # autograd's numpy does not seem to support object arrays.
            # The following snippet works with numpy 1.14.2 but not
            # autograd's numpy commit 5d49ee.
            #
            # foo = OrderedDict(a=5)
            # bar = onp.array([foo for i in range(3)])
            # print(bar[0]['a']) # Gives an index error.
            #
            raise NotImplementedError(
                'PatternArray does not support patterns whose folded ' +
                'values are not numpy.ndarray types.')

        super().__init__(
            num_elements * base_pattern.flat_length(free=False),
            num_elements * base_pattern.flat_length(free=True))

    def __str__(self):
        return('PatternArray {} of {}'.format(
            self.__shape, self.__base_pattern))

    def shape(self):
        return self.__shape

    def base_pattern(self):
        return self.__base_pattern

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if self.__shape != other.shape():
            return False
        if self.__base_pattern != other.base_pattern():
            return False
        return True

    def empty(self, valid):
        empty_pattern = self.__base_pattern.empty(valid=valid)
        repeated_array = np.array([ empty_pattern \
            for item in itertools.product(*self.__array_ranges)])
        return np.reshape(
            repeated_array, self.__shape + self.__folded_pattern_shape)

    # Get a slice for the elements in a vector of length flat_length
    # corresponding to
    # element item of the array, where obs is a tuple indexing into the
    # array of shape self.__shape.
    def _stacked_obs_slice(self, item, flat_length):
        assert len(item) == len(self.__shape)
        linear_item = np.ravel_multi_index(item, self.__shape) * flat_length
        return slice(linear_item, linear_item + flat_length)

    def fold(self, flat_val, free):
        flat_val = np.atleast_1d(flat_val)
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        if flat_val.size != self.flat_length(free):
            error_string = \
                'Wrong size for parameter {}.  Expected {}, got {}'.format(
                    self.name, str(flat_length), str(flat_val.size))
            raise ValueError(error_string)

        flat_length = self.__base_pattern.flat_length(free)
        folded_array = np.array([ \
            self.__base_pattern.fold(
                flat_val[self._stacked_obs_slice(item, flat_length)], free) \
            for item in itertools.product(*self.__array_ranges) ])
        return np.reshape(
            folded_array, self.__shape + self.__folded_pattern_shape)

    def flatten(self, folded_val, free):
        return np.hstack(np.array([ \
            self.__base_pattern.flatten(folded_val[item], free=free) \
                for item in itertools.product(*self.__array_ranges) ]))

    def flat_length(self, free):
        return self._free_flat_length if free else self._flat_length

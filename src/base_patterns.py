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
# Dictionary pattern.

class OrderedDictPattern(Pattern):
    def __init__(self):
        self.__pattern_dict = OrderedDict()
        super().__init__(0, 0)

    def __str__(self):
        pattern_strings = [
            '\t' + key + str(self.__pattern_dict[key]) \
            for key in self.__pattern_dict ]
        return \
            'OrderedDict:\n' + \
            '\n'.join(pattern_strings)

    def __eq__(self, other):
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

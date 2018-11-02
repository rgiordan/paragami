from collections import OrderedDict
import itertools

from scipy.sparse import coo_matrix, block_diag
import autograd.numpy as np

from .base_patterns import Pattern


##########################
# Dictionary of patterns.

class PatternDict(Pattern):
    """
    A dictionary of patterns (which is itself a pattern).

    Methods
    ------------
    lock:
        Prevent additional patterns from being added or removed.

    Examples
    ------------
    .. code-block:: python

        import paragami

        # Add some patterns.
        dict_pattern = paragami.PatternDict()
        dict_pattern['vec'] = paragami.NumericArrayPattern(shape=(2, ))
        dict_pattern['mat'] = paragami.PSDSymmetricMatrixPattern(size=3)

        # Dictionaries can also contain dictionaries (but they have to
        # be populated /before/ being added to the parent).
        sub_dict_pattern = paragami.PatternDict()
        sub_dict_pattern['vec1'] = paragami.NumericArrayPattern(shape=(2, ))
        sub_dict_pattern['vec2'] = paragami.NumericArrayPattern(shape=(2, ))
        dict_pattern['sub_dict'] = sub_dict_pattern

        # We're done adding patterns, so lock the dictionary.
        dict_pattern.lock()

        # Get a random intial value for the whole dictionary.
        dict_val = dict_pattern.random()
        print(dict_val['mat']) # Prints a 3x3 positive definite numpy matrix.

        # Get a flattened value of the whole dictionary.
        dict_val_flat = dict_pattern.flatten(dict_val, free=True)

        # Get a new random folded value of the dictionary.
        new_dict_val_flat = np.random.random(len(dict_val_flat))
        new_dict_val = dict_pattern.fold(new_dict_val_flat, free=True)
    """
    def __init__(self):
        self.__pattern_dict = OrderedDict()

        # __lock determines whether new elements can be added.
        self.__lock = False
        super().__init__(0, 0)

    def lock(self):
        self.__lock = True

    def __str__(self):
        pattern_strings = [
            '\t[' + key + '] = ' + str(self.__pattern_dict[key])
            for key in self.__pattern_dict]
        return \
            'OrderedDict:\n' + \
            '\n'.join(pattern_strings)

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if self.__pattern_dict.keys() != other.keys():
            return False
        for pattern_name in self.__pattern_dict.keys():
            if self.__pattern_dict[pattern_name] != other[pattern_name]:
                return False
        return True

    def __getitem__(self, key):
        return self.__pattern_dict[key]

    def _check_lock(self):
        if self.__lock:
            raise ValueError(
                'The dictionary is locked, and its values cannot be changed.')

    def __setitem__(self, pattern_name, pattern):
        self._check_lock()
        # if pattern_name in self.__pattern_dict.keys():
        #     self.__delitem__(pattern_name)

        self.__pattern_dict[pattern_name] = pattern

        # We cannot allow pattern dictionaries to change their size
        # once they've been included as members in another dictionary,
        # since we have no way of updating the parent dictionary's size.
        # To avoid unexpected errors, lock any dictionary that is set as
        # a member.
        if type(self.__pattern_dict[pattern_name]) is PatternDict:
            self.__pattern_dict[pattern_name].lock()

        self._free_flat_length = self._update_flat_length(free=True)
        self._flat_length = self._update_flat_length(free=False)

    def __delitem__(self, pattern_name):
        self._check_lock()

        pattern = self.__pattern_dict[pattern_name]
        self.__pattern_dict.pop(pattern_name)

        self._free_flat_length = self._update_flat_length(free=True)
        self._flat_length = self._update_flat_length(free=False)

    def keys(self):
        return self.__pattern_dict.keys()

    def empty(self, valid):
        empty_val = OrderedDict()
        for pattern_name, pattern in self.__pattern_dict.items():
            empty_val[pattern_name] = pattern.empty(valid)
        return empty_val

    def fold(self, flat_val, free, validate=None):
        flat_val = np.atleast_1d(flat_val)
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        flat_length = self.flat_length(free)
        if flat_val.size != flat_length:
            error_string = \
                ('Wrong size for pattern dictionary {}.\n' +
                 'Expected {}, got {}.').format(
                    str(self), str(flat_length), str(flat_val.size))
            raise ValueError(error_string)

        # TODO: add an option to do this -- and other operations -- in place.
        folded_val = OrderedDict()
        offset = 0
        for pattern_name, pattern in self.__pattern_dict.items():
            pattern_flat_length = pattern.flat_length(free)
            pattern_flat_val = flat_val[offset:(offset + pattern_flat_length)]
            offset += pattern_flat_length
            folded_val[pattern_name] = pattern.fold(
                pattern_flat_val, free=free, validate=validate)
        return folded_val

    def flatten(self, folded_val, free, validate=None):
        flat_length = self.flat_length(free)
        offset = 0
        flat_val = np.full(flat_length, float('nan'))
        for pattern_name, pattern in self.__pattern_dict.items():
            pattern_flat_length = pattern.flat_length(free)
            flat_val[offset:(offset + pattern_flat_length)] = \
                pattern.flatten(
                    folded_val[pattern_name], free=free, validate=validate)
            offset += pattern_flat_length
        return flat_val

    def _update_flat_length(self, free):
        # This is a little wasteful with the benefit of being less error-prone
        # than adding and subtracting lengths as keys are changed.
        return np.sum([pattern.flat_length(free) for pattern_name, pattern in
                       self.__pattern_dict.items()])

    def unfreeing_jacobian(self, folded_val, sparse=True):
        jacobians = []
        for pattern_name, pattern in self.__pattern_dict.items():
            jac = pattern.unfreeing_jacobian(
                folded_val[pattern_name], sparse=True)
            jacobians.append(jac)

        sp_jac = block_diag(jacobians, format='coo')

        if sparse:
            return sp_jac
        else:
            return np.array(sp_jac.todense())

    def freeing_jacobian(self, folded_val, sparse=True):
        jacobians = []
        for pattern_name, pattern in self.__pattern_dict.items():
            jac = pattern.freeing_jacobian(
                folded_val[pattern_name], sparse=True)
            jacobians.append(jac)

        sp_jac = block_diag(jacobians, format='coo')
        if sparse:
            return sp_jac
        else:
            return np.array(sp_jac.todense())


##########################
# An array of a pattern.

class PatternArray(Pattern):
    """
    An array of a pattern (which is also itself a pattern).

    The first indices of the folded pattern are the array and the final
    indices are of the base pattern.  For example, if `shape=(3, 4)`
    and `base_pattern = PSDSymmetricMatrixPattern(size=5)`, then the folded
    value of the array will have shape `(3, 4, 5, 5)`, where the entry
    `folded_val[i, j, :, :]` is a 5x5 positive definite matrix.

    Currently this can only contain patterns whose folded values are
    numeric arrays (i.e., `NumericArrayPattern`, `SimplexArrayPattern`, and
    `PSDSymmetricMatrixPattern`).

    Methods
    -------------
    shape
        Returns the shape of the entire folded array not including the shape
        of the base pattern.

    base_pattern
        Returns the pattern contained in each element of the array.
    """
    def __init__(self, shape, base_pattern):
        """
        Parameters
        ------------
        shape: tuple of int
            The shape of the array (not including the base parameter)
        base_pattern:
            The base pattern.
        """
        # TODO: change the name shape -> array_shape
        # and have shape be the whole array, including the pattern.
        self.__shape = shape
        self.__array_ranges = [range(0, t) for t in self.__shape]

        num_elements = np.prod(self.__shape)
        self.__base_pattern = base_pattern

        # Check whether the base_pattern takes values that are numpy arrays.
        # If they are, then the unfolded value will be a single numpy array
        # of shape __shape + base_pattern.empty().shape.
        empty_pattern = self.__base_pattern.empty(valid=False)
        if type(empty_pattern) is np.ndarray:
            self.__folded_pattern_shape = empty_pattern.shape
        else:
            # autograd's numpy does not seem to support object arrays.
            # The following snippet works with numpy 1.14.2 but not
            # autograd's numpy (as of commit 5d49ee anyway).
            #
            # >>> import autograd.numpy as np
            # >>> foo = OrderedDict(a=5)
            # >>> bar = np.array([foo for i in range(3)])
            # >>> print(bar[0]['a']) # Gives an index error.
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
        repeated_array = np.array(
            [empty_pattern
             for item in itertools.product(*self.__array_ranges)])
        return np.reshape(
            repeated_array, self.__shape + self.__folded_pattern_shape)

    def _stacked_obs_slice(self, item, flat_length):
        """
        Get the slice in a flat array corresponding to ``item``.

        Parameters
        -------------
        item: tuple
            A tuple of indices into the array of patterns (i.e.,
            into the shape ``__shape``).
        flat_length: integer
            The length of a single flat pattern.

        Returns
        ---------------
        A slice for the elements in a vector of length ``flat_length``
        corresponding to element item of the array, where ``item`` is a tuple
        indexing into the array of shape ``__shape``.
        """
        assert len(item) == len(self.__shape)
        linear_item = np.ravel_multi_index(item, self.__shape) * flat_length
        return slice(linear_item, linear_item + flat_length)

    def fold(self, flat_val, free, validate=None):
        flat_val = np.atleast_1d(flat_val)
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        if flat_val.size != self.flat_length(free):
            error_string = \
                'Wrong size for parameter {}.  Expected {}, got {}'.format(
                    self.name, str(self.flat_length(free)), str(flat_val.size))
            raise ValueError(error_string)

        flat_length = self.__base_pattern.flat_length(free)
        folded_array = np.array([
            self.__base_pattern.fold(
                flat_val[self._stacked_obs_slice(item, flat_length)],
                free=free, validate=validate)
            for item in itertools.product(*self.__array_ranges)])
        return np.reshape(
            folded_array, self.__shape + self.__folded_pattern_shape)

    def flatten(self, folded_val, free, validate=None):
        return np.hstack(np.array([
            self.__base_pattern.flatten(
                folded_val[item], free=free, validate=validate)
            for item in itertools.product(*self.__array_ranges)]))

    def flat_length(self, free):
        return self._free_flat_length if free else self._flat_length

    def unfreeing_jacobian(self, folded_val, sparse=True):
        base_flat_length = self.__base_pattern.flat_length(free=True)
        base_freeflat_length = self.__base_pattern.flat_length(free=True)

        jacobians = []
        for item in itertools.product(*self.__array_ranges):
            jac = self.__base_pattern.unfreeing_jacobian(
                folded_val[item], sparse=True)
            jacobians.append(jac)
        sp_jac = block_diag(jacobians, format='coo')

        if sparse:
            return sp_jac
        else:
            return np.array(sp_jac.todense())

    def freeing_jacobian(self, folded_val, sparse=True):
        base_flat_length = self.__base_pattern.flat_length(free=True)
        base_freeflat_length = self.__base_pattern.flat_length(free=True)

        jacobians = []
        for item in itertools.product(*self.__array_ranges):
            jac = self.__base_pattern.freeing_jacobian(
                folded_val[item], sparse=True)
            jacobians.append(jac)
        sp_jac = block_diag(jacobians, format='coo')

        if sparse:
            return sp_jac
        else:
            return np.array(sp_jac.todense())

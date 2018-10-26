import numpy as np


class FlattenedFunction:
    def __init__(self, original_fun, patterns, free, argnums=None):
        self._fun = original_fun
        self._patterns = np.atleast_1d(patterns)
        if argnums is None:
            argnums = np.arange(0, len(self._patterns))
        if len(self._patterns.shape) != 1:
            raise ValueError('patterns must be a 1d vector.')
        self._argnums = np.atleast_1d(argnums)
        self._argnum_sort = np.argsort(self._argnums)
        self.free = np.broadcast_to(free, self._patterns.shape)

        self._validate_args()

    def _validate_args(self):
        if len(self._argnums.shape) != 1:
            raise ValueError('argnums must be a 1d vector.')
        if len(self._argnums) != len(self._patterns):
            raise ValueError('argnums must be the same length as patterns.')
        if len(self.free.shape) != 1:
            raise ValueError(
                'free must be a single boolean or a 1d vector of booleans.')
        if len(self.free) != len(self._patterns):
            raise ValueError(
                'free must broadcast to the same shape as patterns.')

    def __str__(self):
        return('Function: {}\nargnums: {}\nfree: {}\npatterns: {}'.format(
            self._fun, self._argnums, self.free, self._patterns))

    def __call__(self, *args, **kwargs):
        # Loop through the arguments from beginning to end, replacing
        # parameters with their flattened values.
        new_args = ()
        last_argnum = 0
        for i in self._argnum_sort:
            argnum = self._argnums[i]
            folded_val = \
                self._patterns[i].fold(args[argnum], free=self.free[i])
            new_args += args[last_argnum:argnum] + (folded_val, )
            last_argnum = argnum + 1

        return self._fun(*new_args, **kwargs)

import numpy as np


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
    # def serialize(self):
    #     raise NotImplementedError()
    #
    # def unserialize(self, serialized_val):
    #     raise NotImplementedError()

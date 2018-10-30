import numpy as np


class Pattern(object):
    """
    A pattern for folding and unfolding a parameter.

    Attributes
    ------------

    Methods
    ---------
    __str__(): A string description of the pattern.
    __eq__(): Check two patterns for equality.

    Examples
    ------------
    Todo.
    """
    def __init__(self, flat_length, free_flat_length):
        """
        Parameters
        -----------
        flat_length : int
            The length of a non-free flattened vector.
        free_flat_length : int
            The length of a free flattened vector.
        """
        self._flat_length = flat_length
        self._free_flat_length = free_flat_length

    def __str__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def fold(self, flat_val, free, validate=None):
        """
        Fold a flat value into a parameter.

        Parameters
        -----------
        flat_val: 1-d float array
            The flattened value.
        free: Boolean
            Whether or not the flattened value is a free parameterization.
        validate: Boolean
            Whether to validate that the folded value respects the constraints.
            If None, default either to the pattern's default or, if that is
            unspecified, to True.

        Returns
        ---------
        The parameter value in its original "folded" shape.
        """
        raise NotImplementedError()

    def flatten(self, folded_val, free, validate=None):
        """
        Flatten a folded value into a flat vector.

        Parameters
        -----------
        folded_val
            The parameter in its original "folded" shape.
        free: Boolean
            Whether or not the flattened value is to be in a free
            parameterization.
        validate: Boolean
            Whether to validate that the folded value respects the constraints.
            If None, default either to the pattern's default or, if that is
            unspecified, to True.

        Returns
        ---------
        1-d vector of floats
            The flattened value.
        """
        raise NotImplementedError()

    # Get the size of the flattened version.
    def flat_length(self, free):
        """
        Return the length of the pattern's flattened value.

        Parameters
        -----------
        free: Boolean
            Whether or not the flattened value is to be in a free
            parameterization.

        Returns
        ---------
        int
            The length of the pattern's flattened value.
        """
        if free:
            return self._free_flat_length
        else:
            return self._flat_length

    # Methods to generate valid values.
    def empty(self, valid):
        """
        Return an empty parameter in its "folded" shape.

        Parameters
        -----------
        valid: Boolean
            Whether or folded shape should be filled with valid values.

        Returns
        ---------
        A parameter value in its original "folded" shape.
        """
        raise NotImplementedError()

    def random(self):
        """
        Return an random, valid parameter in its "folded" shape.

        Returns
        ---------
        A random parameter value in its original "folded" shape.
        """
        return self.fold(np.random.random(self._free_flat_length), free=True)

    # These are currently not implemented, but we should.
    # Maybe this should be to / from JSON.
    # def serialize(self):
    #     raise NotImplementedError()
    #
    # def unserialize(self, serialized_val):
    #     raise NotImplementedError()

#!/usr/bin/env python3
import copy
import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np

import collections

import paragami

from autograd.test_util import check_grads


def _test_pattern(testcase, pattern, valid_value,
                  check_equal=assert_array_almost_equal):

    # Execute required methods.
    empty_val = pattern.empty(valid=True)
    pattern.flatten(empty_val, free=False)
    empty_val = pattern.empty(valid=False)

    random_val = pattern.random()
    pattern.flatten(random_val, free=False)

    str(pattern)

    # Make sure to test != using a custom test.
    testcase.assertTrue(pattern == pattern)

    # pattern_serial = pattern.serialize()
    # pattern.unserialize(pattern_serial)

    # Test folding and unfolding.
    for free in [True, False]:
        flat_val = pattern.flatten(valid_value, free=free)
        testcase.assertEqual(len(flat_val), pattern.flat_length(free))
        folded_val = pattern.fold(flat_val, free=free)
        check_equal(valid_value, folded_val)

    # Test the freeing and unfreeing Jacobians.
    for sparse in [True, False]:
        freeing_jac = pattern.freeing_jacobian(valid_value, sparse)
        unfreeing_jac = pattern.unfreeing_jacobian(valid_value, sparse)

class TestPatterns(unittest.TestCase):
    def test_simplex_array_patterns(self):
        def test_shape_and_size(simplex_size, array_shape):
            shape = array_shape + (simplex_size, )
            valid_value = np.random.random(shape) + 0.1
            valid_value = \
                valid_value / np.sum(valid_value, axis=-1, keepdims=True)

            pattern = paragami.SimplexArrayPattern(
                simplex_size, array_shape)
            _test_pattern(self, pattern, valid_value)

        test_shape_and_size(4, (2, 3))
        test_shape_and_size(2, (2, 3))
        test_shape_and_size(2, (2, ))

        self.assertTrue(
            paragami.SimplexArrayPattern(3, (2, 3)) !=
            paragami.SimplexArrayPattern(3, (2, 4)))

        self.assertTrue(
            paragami.SimplexArrayPattern(4, (2, 3)) !=
            paragami.SimplexArrayPattern(3, (2, 3)))

    def test_numeric_array_patterns(self):
        for test_shape in [(1, ), (2, ), (2, 3), (2, 3, 4)]:
            valid_value = np.random.random(test_shape)
            pattern = paragami.NumericArrayPattern(test_shape)
            _test_pattern(self, pattern, valid_value)

            pattern = paragami.NumericArrayPattern(test_shape, lb=-1)
            _test_pattern(self, pattern, valid_value)

            pattern = paragami.NumericArrayPattern(test_shape, ub=2)
            _test_pattern(self, pattern, valid_value)

            pattern = paragami.NumericArrayPattern(test_shape, lb=-1, ub=2)
            _test_pattern(self, pattern, valid_value)

            # Test equality comparisons.
            self.assertTrue(
                paragami.NumericArrayPattern((1, 2)) !=
                paragami.NumericArrayPattern((1, )))

            self.assertTrue(
                paragami.NumericArrayPattern((1, 2)) !=
                paragami.NumericArrayPattern((1, 3)))

            self.assertTrue(
                paragami.NumericArrayPattern((1, 2), lb=2) !=
                paragami.NumericArrayPattern((1, 2)))

            self.assertTrue(
                paragami.NumericArrayPattern((1, 2), lb=2, ub=4) !=
                paragami.NumericArrayPattern((1, 2), lb=2))

            # Check that singletons work.
            pattern = paragami.NumericArrayPattern(shape=(1, ))
            _test_pattern(self, pattern, 1.0)

    def test_psdmatrix_patterns(self):
        dim = 3
        valid_value = np.eye(dim) * 3 + np.full((dim, dim), 0.1)
        pattern = paragami.PSDSymmetricMatrixPattern(dim)
        _test_pattern(self, pattern, valid_value)

        pattern = paragami.PSDSymmetricMatrixPattern(dim, diag_lb=0.5)
        _test_pattern(self, pattern, valid_value)

        self.assertTrue(
            paragami.PSDSymmetricMatrixPattern(3) !=
            paragami.PSDSymmetricMatrixPattern(4))

        self.assertTrue(
            paragami.PSDSymmetricMatrixPattern(3, diag_lb=2) !=
            paragami.PSDSymmetricMatrixPattern(3))
    def test_dictionary_patterns(self):
        def check_dict_equal(dict1, dict2):
            self.assertEqual(dict1.keys(), dict2.keys())
            for key in dict1:
                if type(dict1[key]) is collections.OrderedDict:
                    check_dict_equal(dict1[key], dict2[key])
                else:
                    assert_array_almost_equal(dict1[key], dict2[key])

        dict_pattern = paragami.PatternDict()
        dict_pattern['a'] = \
            paragami.NumericArrayPattern((2, 3, 4), lb=-1, ub=2)
        dict_pattern['b'] = \
            paragami.NumericArrayPattern((5, ), lb=-1, ub=10)
        dict_pattern['c'] = \
            paragami.NumericArrayPattern((5, 2), lb=-1, ub=10)
        subdict = paragami.PatternDict()
        subdict['suba'] = \
            paragami.NumericArrayPattern((2, ))
        dict_pattern['d'] = subdict

        self.assertEqual(list(dict_pattern.keys()), ['a', 'b', 'c', 'd'])

        dict_val = dict_pattern.random()
        _test_pattern(self, dict_pattern, dict_val, check_dict_equal)

        # Check that it works with ordinary dictionaries, not only OrderedDict.
        plain_dict_val = dict(dict_val)
        _test_pattern(self, dict_pattern, plain_dict_val, check_dict_equal)

        # Check deletion and non-equality.
        old_dict_pattern = copy.deepcopy(dict_pattern)
        del dict_pattern['b']
        self.assertTrue(dict_pattern != old_dict_pattern)
        dict_val = dict_pattern.random()
        _test_pattern(self, dict_pattern, dict_val, check_dict_equal)

        # Check adding a new element.
        dict_pattern['d'] = \
            paragami.NumericArrayPattern((4, ), lb=-1, ub=10)
        dict_val = dict_pattern.random()
        _test_pattern(self, dict_pattern, dict_val, check_dict_equal)

        # Check locking
        dict_pattern.lock()

        def delete():
            del dict_pattern['b']

        def add():
            dict_pattern['new'] = \
                paragami.NumericArrayPattern((4, ))

        def modify():
            dict_pattern['a'] = \
                paragami.NumericArrayPattern((4, ))

        self.assertRaises(ValueError, delete)
        self.assertRaises(ValueError, add)
        self.assertRaises(ValueError, modify)

    def test_pattern_array(self):
        array_pattern = paragami.NumericArrayPattern(
            shape=(2, ), lb=-1, ub=10.0)
        pattern_array = paragami.PatternArray((2, 3), array_pattern)
        valid_value = pattern_array.random()
        _test_pattern(self, pattern_array, valid_value)

        matrix_pattern = paragami.PSDSymmetricMatrixPattern(size=2)
        pattern_array = paragami.PatternArray((2, 3), matrix_pattern)
        valid_value = pattern_array.random()
        _test_pattern(self, pattern_array, valid_value)

        self.assertTrue(
            paragami.PatternArray((3, 3), matrix_pattern) !=
            paragami.PatternArray((2, 3), matrix_pattern))

        self.assertTrue(
            paragami.PatternArray((2, 3), array_pattern) !=
            paragami.PatternArray((2, 3), matrix_pattern))


class TestHelperFunctions(unittest.TestCase):
    def _test_logsumexp(self, mat, axis):
        # Test the more numerically stable version with this simple
        # version of logsumexp.
        def logsumexp_simple(mat, axis):
            return np.log(np.sum(np.exp(mat), axis=axis, keepdims=True))

        check_grads(
            paragami.simplex_patterns._logsumexp,
            modes=['fwd', 'rev'], order=3)(mat, axis)

        assert_array_almost_equal(
            logsumexp_simple(mat, axis),
            paragami.simplex_patterns._logsumexp(mat, axis))


    def test_logsumexp(self):
        mat = np.random.random((3, 3, 3))
        self._test_logsumexp(mat, 0)

    def test_pdmatrix_custom_autodiff(self):
        x_vec = np.random.random(6)
        x_mat = paragami.psdmatrix_patterns._unvectorize_ld_matrix(x_vec)

        check_grads(
            paragami.psdmatrix_patterns._vectorize_ld_matrix,
            modes=['fwd', 'rev'], order=3)(x_mat)
        check_grads(
            paragami.psdmatrix_patterns._unvectorize_ld_matrix,
            modes=['fwd', 'rev'], order=3)(x_vec)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
import copy
import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np

import array_patterns
import base_patterns


def _test_pattern(
    testcase, pattern, valid_value, check_equal=assert_array_almost_equal):

    # Execute required methods.
    empty_val = pattern.empty(valid=True)
    pattern.flatten(empty_val, free=False)
    empty_val = pattern.empty(valid=False)

    random_val = pattern.random()
    pattern.flatten(random_val, free=False)

    str(pattern)

    # Make sure to test != using a custom test.
    testcase.assertTrue(pattern == pattern)

    #pattern_serial = pattern.serialize()
    #pattern.unserialize(pattern_serial)

    for free in [True, False]:
        flat_val = pattern.flatten(valid_value, free=free)
        testcase.assertEqual(len(flat_val), pattern.flat_length(free))
        folded_val = pattern.fold(flat_val, free=free)
        check_equal(valid_value, folded_val)


class TestPatterns(unittest.TestCase):
    def test_array_patterns(self):
        for test_shape in [(2, ), (2, 3), (2, 3, 4)]:
            valid_value = np.random.random(test_shape)
            pattern = array_patterns.ArrayPattern(test_shape)
            _test_pattern(self, pattern, valid_value)

            pattern = array_patterns.ArrayPattern(test_shape, lb=-1)
            _test_pattern(self, pattern, valid_value)

            pattern = array_patterns.ArrayPattern(test_shape, ub=2)
            _test_pattern(self, pattern, valid_value)

            pattern = array_patterns.ArrayPattern(test_shape, lb=-1, ub=2)
            _test_pattern(self, pattern, valid_value)

            # Test equality comparisons.
            self.assertTrue(
                array_patterns.ArrayPattern((1, 2)) !=
                array_patterns.ArrayPattern((1, )))

            self.assertTrue(
                array_patterns.ArrayPattern((1, 2)) !=
                array_patterns.ArrayPattern((1, 3)))

            self.assertTrue(
                array_patterns.ArrayPattern((1, 2), lb=2) !=
                array_patterns.ArrayPattern((1, 2)))

            self.assertTrue(
                array_patterns.ArrayPattern((1, 2), lb=2, ub=4) !=
                array_patterns.ArrayPattern((1, 2), lb=2))


    def test_dictionary_patterns(self):
        def check_dict_equal(dict1, dict2):
            self.assertEqual(dict1.keys(), dict2.keys())
            for key  in dict1:
                assert_array_almost_equal(dict1[key], dict2[key])

        dict_pattern = base_patterns.OrderedDictPattern()
        dict_pattern['a'] = \
            array_patterns.ArrayPattern((2, 3, 4), lb=-1, ub=2)
        dict_pattern['b'] = \
            array_patterns.ArrayPattern((5, ), lb=-1, ub=10)
        dict_pattern['c'] = \
            array_patterns.ArrayPattern((5, 2), lb=-1, ub=10)

        self.assertEqual(list(dict_pattern.keys()), ['a', 'b', 'c'])

        dict_val = dict_pattern.random()
        _test_pattern(self, dict_pattern, dict_val, check_dict_equal)

        old_dict_pattern = copy.deepcopy(dict_pattern)
        del dict_pattern['b']
        self.assertTrue(dict_pattern != old_dict_pattern)
        dict_val = dict_pattern.random()
        _test_pattern(self, dict_pattern, dict_val, check_dict_equal)

        dict_pattern['d'] = \
            array_patterns.ArrayPattern((4, ), lb=-1, ub=10)
        dict_val = dict_pattern.random()
        _test_pattern(self, dict_pattern, dict_val, check_dict_equal)


if __name__ == '__main__':
    unittest.main()

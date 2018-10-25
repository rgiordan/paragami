#!/usr/bin/env python3
import copy
import unittest
import numpy
from numpy.testing import assert_array_almost_equal

import autograd
import autograd.numpy as np

import paragami
from function_patterns import FlattenedFunction


class TestPatterns(unittest.TestCase):
    def test_flatten_function(self):
        pattern = paragami.OrderedDictPattern()
        pattern['a'] = paragami.NumericArrayPattern((2, 3, 4), lb=-1, ub=2)
        pattern['b'] = paragami.NumericArrayPattern((5, ), lb=-1, ub=10)
        param_val = pattern.random()
        x = 3
        y = 4
        z = 5

        def tf1(param_val):
            return np.mean(param_val['a'] ** 2) + np.mean(param_val['b'] ** 2)

        def tf2(x, param_val, y=5):
            return \
                np.mean(param_val['a'] ** 2) + \
                np.mean(param_val['b'] ** 2) + x**2 + y**2

        def tf3(a, b):
            return np.mean(a**2) + np.mean(b**2)

        def tf4(x, a, z, b, y=5):
            return np.mean(a**2) + np.mean(b**2) + x**2 + y**2 + z**2

        for free in [True, False]:
            param_val_flat = pattern.flatten(param_val, free=free)

            tf1_flat = FlattenedFunction(tf1, pattern, free)
            assert_array_almost_equal(
                tf1(param_val), tf1_flat(param_val_flat))

            tf2_flat = FlattenedFunction(tf2, pattern, free, argnums=1)
            assert_array_almost_equal(
                tf2(x, param_val, y=y), tf2_flat(x, param_val_flat, y=y))

            a_flat = pattern['a'].flatten(param_val['a'], free=free)
            b_flat = pattern['b'].flatten(param_val['b'], free=free)

            # Check when both arguments are free
            tf3_flat = FlattenedFunction(
                tf3, [ pattern['a'], pattern['b'] ], free)
            assert_array_almost_equal(
                tf3(param_val['a'], param_val['b']),
                tf3_flat(a_flat, b_flat))

            tf4_flat = FlattenedFunction(
                tf4, [ pattern['a'], pattern['b'] ], free, argnums=[1, 3])
            assert_array_almost_equal(
                tf4(x, param_val['a'], z, param_val['b'], y=y),
                tf4_flat(x, a_flat, z, b_flat, y=y))

            # Check when the arguments differ in whether they are free
            a_flat = pattern['a'].flatten(param_val['a'], free=free)
            b_flat = pattern['b'].flatten(param_val['b'], free=not free)

            tf3_flat = FlattenedFunction(
                tf3, [ pattern['a'], pattern['b'] ], [free, not free])
            assert_array_almost_equal(
                tf3(param_val['a'], param_val['b']),
                tf3_flat(a_flat, b_flat))

            tf4_flat = FlattenedFunction(
                tf4, [ pattern['a'], pattern['b'] ],
                [free, not free], argnums=[1, 3])
            assert_array_almost_equal(
                tf4(x, param_val['a'], z, param_val['b'], y=y),
                tf4_flat(x, a_flat, z, b_flat, y=y))


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

import unittest
from numpy.testing import assert_array_almost_equal

import autograd.numpy as np
from autograd.test_util import check_grads

import itertools

import paragami


def get_test_pattern():
    # autograd will pass invalid values, so turn off value checking.
    pattern = paragami.PatternDict()
    pattern['array'] = paragami.NumericArrayPattern(
        (2, 3, 4), lb=-1, ub=2, default_validate=False)
    pattern['mat'] = paragami.PSDSymmetricMatrixPattern(
        3, default_validate=False)
    pattern['simplex'] = paragami.SimplexArrayPattern(
        2, (3, ), default_validate=False)
    subdict = paragami.PatternDict()
    subdict['array2'] = paragami.NumericArrayPattern(
        (2, ), lb=-3, ub=5, default_validate=False)
    pattern['dict'] = subdict

    return pattern

def get_small_test_pattern():
    # autograd will pass invalid values, so turn off value checking.
    pattern = paragami.PatternDict()
    pattern['array'] = paragami.NumericArrayPattern(
        (2, 3, 4), lb=-1, ub=2, default_validate=False)
    pattern['mat'] = paragami.PSDSymmetricMatrixPattern(
        3, default_validate=False)

    return pattern


def assert_test_dict_equal(d1, d2):
    """Assert that dictionaries corresponding to test pattern are equal.
    """
    for k in ['array', 'mat', 'simplex']:
        assert_array_almost_equal(d1[k], d2[k])
    assert_array_almost_equal(d1['dict']['array2'], d2['dict']['array2'])


# Test functions that work with get_test_pattern() or
# get_small_test_pattern().
def fold_to_num(param_folded):
    return \
        np.mean(param_folded['array'] ** 2) + \
        np.mean(param_folded['mat'] ** 2)

def flat_to_num(param_flat, pattern, free):
    param_folded = pattern.fold(param_flat, free=free)
    return fold_to_num(param_folded)

def num_to_fold(x):
    pattern = get_test_pattern()
    new_param = pattern.get_empty(valid=True)
    new_param['array'] = param_folded['array'] + x
    new_param['mat'] = x * param_folded['mat']
    return new_param

def num_to_flat(x, pattern, free):
    new_param = num_to_fold(x)
    return pattern.flatten(new_param, free=free)


class TestFlatteningAndFolding(unittest.TestCase):
    def _test_transform_input(
        self, original_fun, patterns, free, argnums, original_is_flat,
        folded_args, flat_args, kwargs):

        orig_args = flat_args if original_is_flat else folded_args
        trans_args = folded_args if original_is_flat else flat_args
        fun_trans = paragami.TransformFunctionInput(
            original_fun, patterns, free,
            original_is_flat, argnums)

        argnums_array = np.atleast_1d(argnums)
        patterns_array = np.atleast_1d(patterns)
        free_array = np.atleast_1d(free)

        # Check that the flattened and original function are the same.
        assert_array_almost_equal(
            original_fun(*orig_args, **kwargs),
            fun_trans(*trans_args, **kwargs))

        # Check that the string method works.
        str(fun_trans)

    # def _test_flatten_function(self, original_fun, patterns, free, argnums,
    #                            args, flat_args, kwargs):
    #
    #     fun_flat = paragami.FlattenFunctionInput(
    #         original_fun, patterns, free, argnums)
    #
    #     # Sanity check that the flat_args were set correctly.
    #     argnums_array = np.atleast_1d(argnums)
    #     patterns_array = np.atleast_1d(patterns)
    #     free_array = np.atleast_1d(free)
    #     for i in range(len(argnums_array)):
    #         argnum = argnums_array[i]
    #         pattern = patterns_array[i]
    #         assert_array_almost_equal(
    #             flat_args[argnum],
    #             pattern.flatten(args[argnum], free=free_array[i]))
    #
    #     # Check that the flattened and original function are the same.
    #     assert_array_almost_equal(
    #         original_fun(*args, **kwargs),
    #         fun_flat(*flat_args, **kwargs))
    #
    #     # Check that the string method works.
    #     str(fun_flat)



    def test_transform_input(self):
        pattern = get_test_pattern()
        param_val = pattern.random()
        x = 3
        y = 4
        z = 5

        def testfun1(param_val):
            return \
                np.mean(param_val['array'] ** 2) + \
                np.mean(param_val['mat'] ** 2)

        def testfun2(x, param_val, y=5):
            return \
                np.mean(param_val['array'] ** 2) + \
                np.mean(param_val['mat'] ** 2) + x**2 + y**2

        def testfun3(param_val, x, y=5):
            return \
                np.mean(param_val['array'] ** 2) + \
                np.mean(param_val['mat'] ** 2) + x**2 + y**2

        ft = [False, True]
        for free, origflat in itertools.product(ft, ft):
            def this_flat_to_num(x):
                return flat_to_num(x, pattern, free)

            param_flat = pattern.flatten(param_val, free=free)
            tf1 = this_flat_to_num if origflat else fold_to_num

            def tf2(x, val, y=5):
                return tf1(val) + x ** 2 + y ** 2

            def tf3(val, x, y=5):
                return tf1(val) + x ** 2 + y ** 2

            self._test_transform_input(
                original_fun=tf1, patterns=pattern, free=free,
                argnums=0,
                original_is_flat=origflat,
                folded_args=(param_val, ),
                flat_args=(param_flat, ),
                kwargs={})

            self._test_transform_input(
                original_fun=tf2, patterns=pattern, free=free,
                argnums=1,
                original_is_flat=origflat,
                folded_args=(x, param_val, ),
                flat_args=(x, param_flat, ),
                kwargs={'y': 5})

            self._test_transform_input(
                original_fun=tf3, patterns=pattern, free=free,
                argnums=0,
                original_is_flat=origflat,
                folded_args=(param_val, x, ),
                flat_args=(param_flat, x, ),
                kwargs={'y': 5})

            # Test once with arrays.
            self._test_transform_input(
                original_fun=tf3, patterns=[pattern], free=[free],
                argnums=[0],
                original_is_flat=origflat,
                folded_args=(param_val, x, ),
                flat_args=(param_flat, x, ),
                kwargs={'y': 5})

            # Test bad inits
            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionInput(
                    tf1, [[ pattern ]], free, origflat, 0)

            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionInput(
                    tf1, pattern, free, origflat, [[0]])

            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionInput(
                    tf1, pattern, free, origflat, [0, 0])

            with self.assertRaises(ValueError):
                fun_flat = paragami.FlattenFunctionInput(
                    tf1, pattern, free, [0, 1])

        # Test two-parameter flattening.
        def scalarfun(x, y, z):
            return x**2 + 2 * y**2 + 3 * z**2

        pattern0 = get_test_pattern()
        pattern1 = get_small_test_pattern()
        param0_val = pattern0.random()
        param1_val = pattern1.random()
        for (free0, free1, origflat) in itertools.product(ft, ft, ft):

            if origflat:
                def tf1(p0, p1):
                    return flat_to_num(p0, pattern0, free0) + \
                           flat_to_num(p1, pattern1, free1)
            else:
                def tf1(p0, p1):
                    return fold_to_num(p0) + fold_to_num(p1)

            def tf2(x, p0, z, p1, y=5):
                return tf1(p0, p1) + scalarfun(x, y, z)

            def tf3(p0, z, p1, x, y=5):
                return tf1(p0, p1) + scalarfun(x, y, z)

            param0_flat = pattern0.flatten(param0_val, free=free0)
            param1_flat = pattern1.flatten(param1_val, free=free1)

            self._test_transform_input(
                original_fun=tf1,
                patterns=[pattern0, pattern1],
                free=[free0, free1],
                argnums=[0, 1],
                original_is_flat=origflat,
                folded_args=(param0_val, param1_val),
                flat_args=(param0_flat, param1_flat),
                kwargs={})

            # Test switching the order of the patterns.
            self._test_transform_input(
                original_fun=tf1,
                patterns=[pattern1, pattern0],
                free=[free1, free0],
                argnums=[1, 0],
                original_is_flat=origflat,
                folded_args=(param0_val, param1_val),
                flat_args=(param0_flat, param1_flat),
                kwargs={})

            self._test_transform_input(
                original_fun=tf2,
                patterns=[pattern1, pattern0],
                free=[free1, free0],
                argnums=[3, 1],
                original_is_flat=origflat,
                folded_args=(x, param0_val, z, param1_val, ),
                flat_args=(x, param0_flat, z, param1_flat),
                kwargs={'y': 5})

            self._test_transform_input(
                original_fun=tf3,
                patterns=[pattern1, pattern0],
                free=[free1, free0],
                argnums=[2, 0],
                original_is_flat=origflat,
                folded_args=(param0_val, z, param1_val, x, ),
                flat_args=(param0_flat, z, param1_flat, x),
                kwargs={'y': 5})


    def test_fold_function_output(self):
        pattern = get_test_pattern()
        param_val = pattern.random()
        param_flat = pattern.flatten(param_val, free=False)
        param_free = pattern.flatten(param_val, free=True)

        def get_param(a, b=0.1):
            param_val = pattern.empty(valid=False)
            param_val['array'][:] = a + b
            param_val['mat'] = \
                a * np.eye(param_val['mat'].shape[0]) + b
            param_val['simplex'] = np.full(param_val['simplex'].shape, 0.5)
            param_val['dict']['array2'][:] = a + b

            return param_val

        for free in [False, True]:
            def get_flat_param(a, b=0.1):
                return pattern.flatten(get_param(a, b=b), free=free)

            get_folded_param = paragami.FoldFunctionOutput(
                get_flat_param, pattern=pattern, free=free)
            a = 0.1
            b = 0.2
            assert_test_dict_equal(
                get_param(a, b=b), get_folded_param(a, b=b))

    def test_flatten_and_fold(self):
        pattern = get_test_pattern()
        pattern_val = pattern.random()
        free_val = pattern.flatten(pattern_val, free=True)

        def operate_on_free(free_val, a, b=2):
            return free_val * a + b

        a = 2
        b = 3

        folded_fun = paragami.FoldFunctionInputAndOutput(
            original_fun=operate_on_free,
            input_patterns=pattern,
            input_free=True,
            input_argnums=0,
            output_pattern=pattern,
            output_free=True)

        pattern_out = folded_fun(pattern_val, a, b=b)
        pattern_out_test = pattern.fold(
            operate_on_free(free_val, a, b=b), free=True)
        assert_test_dict_equal(pattern_out_test, pattern_out)

    def test_autograd(self):
        pattern = get_test_pattern()

        # The autodiff tests produces non-symmetric matrices.
        pattern['mat'].default_validate = False
        param_val = pattern.random()

        def tf1(param_val):
            return \
                np.mean(param_val['array'] ** 2) + \
                np.mean(param_val['mat'] ** 2)

        for free in [True, False]:
            tf1_flat = paragami.FlattenFunctionInput(tf1, pattern, free)
            param_val_flat = pattern.flatten(param_val, free=free)
            check_grads(
                tf1_flat, modes=['rev', 'fwd'], order=2)(param_val_flat)


if __name__ == '__main__':
    unittest.main()

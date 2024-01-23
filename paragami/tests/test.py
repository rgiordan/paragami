#!/usr/bin/env python3

import paragami
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal
import unittest

class TestParagami(unittest.TestCase):
    x = jnp.array([1.0])
    assert_array_almost_equal(x, x)

if __name__ == '__main__':
    unittest.main()

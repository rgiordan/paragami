I'm upgrading an old project from `autograd` to `jax`  and finding `jax` to be many times slower than `numpy` on the CPU.  (The performance of `autograd` was comparable to that of `numpy`.)

To demonstrate this, I copied the benchmarking script from https://github.com/google/jax/issues/4134 and ran a few other core numpy functions.  My slowdown is much more severe than that reported in https://github.com/google/jax/issues/4135.  I'm new to `jax`, so I apologize in advance if I misunderstood some aspect of the benchmarking script (or of `jax` in general).

The results of the below script are:
```
Jax version	 0.1.75
Numpy version	 1.18.1
Python version	 3.6.9 (default, Jul 17 2020, 12:50:27)
[GCC 8.4.0]
/home/rgiordan/.local/lib/python3.6/site-packages/jax/lib/xla_bridge.py:130: UserWarning: No GPU/TPU found, falling back to CPU.
  warnings.warn('No GPU/TPU found, falling back to CPU.')

=====================
cumsum timing
jax: 0.768s
np:  0.028s

=====================
sum timing
jax: 0.490s
np:  0.021s

=====================
hstack timing
jax: 0.999s
np:  0.064s

=====================
atleast_1d timing
jax: 0.220s
np:  0.004s

=====================
all timing
jax: 0.489s
np:  0.027s
```

Script:

```python
# Usage:
# taskset -c 0 python3 ./jax_vs_np_benchmark.py

import jax
import numpy as np
import jax.numpy as jnp
import sys
import time
jax.config.update('jax_enable_x64', True)

print('Jax version\t', jax.__version__)
print('Numpy version\t', np.__version__)
print('Python version\t', sys.version)


def benchmark_fun(desc, jax_fun, np_fun, input, num_evals):
    print(f'\n=====================\n{desc} timing')
    res_jax = jax_fun(input).block_until_ready()
    start = time.perf_counter()
    for _ in range(num_evals):
        jax_fun(input).block_until_ready()
    end = time.perf_counter()
    print(f'jax: {end - start:.3f}s')

    input_np = np.array(input)
    res_np = np_fun(input_np)
    start = time.perf_counter()
    for _ in range(num_evals):
        np_fun(input_np)
    end = time.perf_counter()
    print(f'np:  {end - start:.3f}s')

    np.testing.assert_allclose(res_jax, res_np)


if __name__ == '__main__':
    key = jax.random.PRNGKey(17)

    arr = jax.random.uniform(key, shape=(360, 3), dtype='float64')

    arrs = [ jax.random.uniform(key, shape=(100, ), dtype='float64')
             for _ in range(10) ]

    ##################
    # cumsum

    @jax.jit
    def cumsum_jax(arr):
        return jnp.cumsum(arr, axis=0)

    def cumsum_np(arr):
        return np.cumsum(arr, axis=0)

    benchmark_fun('cumsum', cumsum_jax, cumsum_np, arr, 5000)


    ##################
    # sum

    @jax.jit
    def sum_jax(arr):
        return jnp.sum(arr)

    benchmark_fun('sum', sum_jax, np.sum, arr, 5000)


    ##################
    # hstack

    @jax.jit
    def hstack_jax(arrs):
        return jnp.hstack(arrs)

    benchmark_fun('hstack', hstack_jax, np.hstack, arrs, 5000)

    ##################
    # atleast_1d

    @jax.jit
    def atleast_1d_jax(arr):
        return jnp.atleast_1d(arr)

    benchmark_fun('atleast_1d', atleast_1d_jax, np.atleast_1d, arr, 5000)

    ##################
    # all

    @jax.jit
    def all_pos_jax(arr):
        return jnp.all(arr > 0.)

    def all_pos_np(arr):
        return np.all(arr > 0.)

    benchmark_fun('all', all_pos_jax, all_pos_np, arr, 5000)
```

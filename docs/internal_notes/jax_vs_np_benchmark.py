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

    arr = jax.random.uniform(key, shape=(3600, 10), dtype='float64')

    big_arr = jax.random.uniform(key, shape=(3600, 30, 10), dtype='float64')

    arrs = [ jax.random.uniform(key, shape=(1000, 10), dtype='float64')
             for _ in range(10) ]

    num_ops = 10

    ##################
    # cumsum

    @jax.jit
    def cumsum_jax(arr):
        return jnp.cumsum(arr, axis=0)

    def cumsum_np(arr):
        return np.cumsum(arr, axis=0)

    benchmark_fun('cumsum', cumsum_jax, cumsum_np, arr, num_ops)


    ##################
    # sum

    @jax.jit
    def sum_jax(arr):
        return jnp.sum(arr)

    benchmark_fun('sum', sum_jax, np.sum, big_arr, num_ops)

    ##################
    # hstack

    @jax.jit
    def hstack_jax(arrs):
        return jnp.hstack(arrs)

    benchmark_fun('hstack', hstack_jax, np.hstack, arrs, num_ops)

    ##################
    # atleast_1d

    @jax.jit
    def atleast_1d_jax(arr):
        return jnp.atleast_1d(arr)

    benchmark_fun('atleast_1d', atleast_1d_jax, np.atleast_1d, big_arr, num_ops)

    ##################
    # all

    @jax.jit
    def all_pos_jax(arr):
        return jnp.all(arr > 0.)

    def all_pos_np(arr):
        return np.all(arr > 0.)

    benchmark_fun('all', all_pos_jax, all_pos_np, arr, 50)

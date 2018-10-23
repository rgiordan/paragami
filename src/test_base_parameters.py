

import LinearResponseVariationalBayes as lrvb
import LinearResponseVariationalBayes.base_parameters as paragami

def check_equal(a, b, tol=1e-9):
    return np.linalg.norm(a - b) < tol


test_shape = (2, 3, 4)
a = np.random.random(test_shape)

array_pattern = paragami.ArrayPattern('a', test_shape, lb=-1, ub=2)

a_flat = array_pattern.flatten(a, free=False)
print(a_flat)
a_fold = array_pattern.fold(a_flat, free=False)
assert check_equal(a, a_fold)

a_flat = array_pattern.flatten(a, free=True)
print(a_flat)
a_fold = array_pattern.fold(a_flat, free=True)
assert check_equal(a, a_fold)

dict_pattern = paragami.OrderedDictPattern('dict')

dict_pattern['a'] = paragami.ArrayPattern('a', (2, 3, 4), lb=-1, ub=2)
dict_pattern['b'] = paragami.ArrayPattern('b', (5, ), lb=-1, ub=10)

dict_val = dict_pattern.random()
print(dict_val['a'])
print(dict_val['b'])

for free in [True, False]:
    dict_flat = dict_pattern.flatten(dict_val, free=free)
    dict_fold = dict_pattern.fold(dict_flat, free=free)
    for key  in dict_fold:
        assert check_equal(dict_fold[key], dict_val[key])

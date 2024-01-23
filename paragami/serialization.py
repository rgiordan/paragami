
import gzip
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import json
import pickle


def ToJSON(x):
    if jnp.isscalar(x):
        return json.dumps(x)
    else:
        return json.dumps(x.tolist())

def FromJSON(x_json):
    x = json.loads(x_json)
    if jnp.isscalar(x):
        return x
    else:
        return jnp.array(x)

def SerializePytree(par):
    par_json = jtu.tree_map(ToJSON, par)
    return json.dumps(par_json)
    
def DeserializePytree(par_json):
    return jtu.tree_map(FromJSON, json.loads(par_json))

def SavePytree(par, filename):
    with gzip.open(filename, 'w') as fout:
        fout.write(SerializePytree(par).encode('utf-8'))
        
def LoadPytree(filename):
    with gzip.open(filename, 'r') as fin:
        par_str = fin.read().decode('utf-8')
    return DeserializePytree(par_str)



# Not sure this is too useful given the above.

# https://docs.python.org/3/library/pickle.html#restricting-globals
class RestrictedUnpickler(pickle.Unpickler):
    safe_types = {
        'HMCState.blackjax.mcmc': [ 'blackjax.mcmchmc' ],
        'jax._src.array': [ '_reconstruct_array' ],
        'blackjax.mcmc.hmc': ['HMCState'],
        'numpy.core.multiarray': ['_reconstruct'],
        'numpy': ['ndarray', 'dtype'],
    }

    def find_class(self, module, name):
        # Only allow safe classes from builtins.
        if name in self.safe_types.get(module, []):
            if module == 'HMCState.blackjax.mcmc':
                return getattr(HMCState.blackjax.mcmc, name)
            elif module == 'jax._src.array':
                return getattr(jax._src.array, name)
            elif module == 'blackjax.mcmc.hmc':
                return getattr(blackjax.mcmc.hmc, name)
            elif module == 'numpy.core.multiarray':
                return getattr(numpy.core.multiarray, name)
            elif module == 'numpy':
                return getattr(numpy, name)
            else:
                raise KeyError(f'module {module} missing from `safe_types`')
        # Forbid everything else.
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" %
                                     (module, name))


def SafeUnpickleHMCState(s):
    """Helper function analogous to pickle.loads()."""
    return RestrictedUnpickler(io.BytesIO(s)).load()

from paragami.pattern_containers import PatternDict, PatternArray
from paragami.numeric_array_patterns import NumericArrayPattern
from paragami.psdmatrix_patterns import PSDSymmetricMatrixPattern
from paragami.function_patterns import FlattenedFunction, Functor
from paragami.simplex_patterns import SimplexArrayPattern
from paragami.optimization_lib import \
    HyperparameterSensitivityLinearApproximation, PreconditionedFunction

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

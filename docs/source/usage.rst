====================
API Documentation
====================

.. Pattern parent class
.. ========================
..
.. Every pattern described herin inherits from the
.. parent `Pattern` class and implements its methods.
..
.. .. autoclass:: paragami.base_patterns.Pattern
..    :members:

Numeric patterns
========================

Numeric arrays
---------------------------
.. autoclass:: paragami.numeric_array_patterns.NumericArrayPattern
   :members:


Positive definite matrices
---------------------------

.. autoclass:: paragami.pdmatrix_patterns.PDMatrixPattern
   :members:

Simplexes
---------------------------

.. autoclass:: paragami.simplex_patterns.SimplexArrayPattern
  :members:

Containers of patterns
========================

Containers of patterns are themselves patterns, and so can
contain instantiations of themselves.

Dictionaries of patterns
---------------------------
.. autoclass:: paragami.pattern_containers.PatternDict
  :members:


Arrays of patterns
---------------------------
.. autoclass:: paragami.pattern_containers.PatternArray
  :members:



Function wrappers
========================

Arrays of patterns
---------------------------
.. autoclass:: paragami.function_patterns.FlattenedFunction
  :members:

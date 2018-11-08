Converting to and from JSON
================================

You can convert a particular pattern class to and from JSON using the
``to_json`` and ``from_json`` methods.

>>> pattern = paragami.NumericArrayPattern(shape=(2, 3))
>>>
>>> # ``pattern_json_string`` is a JSON string that can be written to a file.
>>> pattern_json_string = pattern.to_json()
>>>
>>> # ``same_pattern`` is identical to ``pattern``.
>>> same_pattern = paragami.NumericArrayPattern.from_json(pattern_json_string)

However, in order to use ``from_json``, you need to know which pattern
the JSON string was generated from.  In order to decode generic JSON strings,
one can use ``get_pattern_from_json``.

>>> pattern_json_string = pattern.to_json()
>>> # ``same_pattern`` is identical to ``pattern``.
>>> same_pattern = paragami.get_pattern_from_json(pattern_json_string)

Before a pattern can be used with
``get_pattern_from_json``, it needs to be registered with
``register_pattern_json``.  All the patterns in ``paragami`` are automatically
registered, but if you define your own patterns they will have to
be registered before they can be used with ``get_pattern_from_json``.

.. autofunction:: paragami.pattern_containers.get_pattern_from_json

.. autofunction:: paragami.pattern_containers.register_pattern_json

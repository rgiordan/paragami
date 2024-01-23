# "Parameter origami": `paragami`.

[![image](https://travis-ci.org/rgiordan/paragami.svg?branch=master)](https://travis-ci.org/rgiordan/paragami)
[![image](https://codecov.io/gh/rgiordan/paragami/branch/master/graph/badge.svg)](https://codecov.io/gh/rgiordan/paragami)

## Description.

Much of the old `paragami` has been made obsolete by developments in `jax`
and tensorflow probability, particularly PyTrees.

However, there is still some useful functionality here.  I'm particularly
keen to 
  - Easily manage and differentiate my own bijectors
  - Easily wrap function input and output, including preconditioning
  - Have safe, boilerplate serialization

I'll update for a more modern era in this branch.



\*  Thanks to Stéfan van der Walt for the suggesting the package name.

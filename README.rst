================================
"Parameter origami": `paragami`
================================

Description
---------------

This is a library (very much still in development) intended to make sensitivity
analysis easier for optimization problems. This package provides tools for the
folding and unfolding collections of parameters for optimization and sensitivity
analysis.

For some background and motivations, see some of our papers:

Covariances, Robustness, and Variational Bayes
Ryan Giordano, Tamara Broderick, Michael I. Jordan
https://arxiv.org/abs/1709.02536

A Swiss Army Infinitesimal Jackknife
Ryan Giordano, Will Stephenson, Runjing Liu, Michael I. Jordan, Tamara Broderick
https://arxiv.org/abs/1806.00550

Evaluating Sensitivity to the Stick Breaking Prior in Bayesian Nonparametrics
Runjing Liu, Ryan Giordano, Michael I. Jordan, Tamara Broderick
https://arxiv.org/abs/1810.06587


Examples
-------------

For some starting examples, see the Jupyter notebooks in
`docs/example_notebooks/`.


Installation
-----------------

To install, run the follwing command, pointing to the root of the git repo:

```sudo -H pip3 install --user -e paragami```.

.. Note that if you do not install with ```--user``` you will have to manually
   remove the egg info in order to re-install.
   ```paragami/paragami.egg-info```
   ```/usr/local/lib/python3.5/dist-packages/paragami.egg-link```

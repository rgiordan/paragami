
.. code:: ipython3

    import paragami
    
    import autograd
    from autograd import numpy as np
    
    # Use the original scipy for functions we don't need to differentiate.
    import scipy as osp

Step 1: Define a model.
-----------------------

For illustration, let’s consider a simple example: a Gaussian maximum
likelihood estimator.

.. math::


   x_n \overset{iid}\sim \mathcal{N}(\mu, \Sigma)\textrm{, for }n=1,...,N.

Let :math:`X = (x_1, ..., x_N)`. We will minimize the loss

.. math::


   \ell(X, \mu, \Sigma) = \frac{1}{2}\sum_{n=1}^N \left((x_n - \mu)^T \Sigma^{-1} (x_n - \mu) + \log |\Sigma| \right).

Specify parameters and draw data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    np.random.seed(42)
    
    num_obs = 1000
    
    # True values of parameters
    true_sigma = \
        np.eye(3) * np.diag(np.array([1, 2, 3])) + \
        np.random.random((3, 3)) * 0.1
    true_sigma = 0.5 * (true_sigma + true_sigma.T)
    
    true_mu = np.array([0, 1, 2])
    
    # Data
    x = np.random.multivariate_normal(
        mean=true_mu, cov=true_sigma, size=(num_obs, ))

Write out the log likelihood and use it to specify a loss function.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def get_normal_log_prob(x, sigma, mu):
        sigma_inv = np.linalg.inv(sigma)
        sigma_det_sign, sigma_log_det = np.linalg.slogdet(sigma)
        if sigma_det_sign <= 0:
            return np.full(float('inf'), x.shape[0])
        else:
            x_centered = x - np.expand_dims(mu, axis=0)
            return -0.5 * (
                np.einsum('ni,ij,nj->n', x_centered, sigma_inv, x_centered) + \
                sigma_log_det)
    
    # The loss function uses the data x from the global scope.
    def get_loss(norm_param_dict):
        return np.sum(
            -1 * get_normal_log_prob(
                x, norm_param_dict['sigma'], norm_param_dict['mu']))
    
    true_norm_param_dict = dict()
    true_norm_param_dict['sigma'] = true_sigma
    true_norm_param_dict['mu'] = true_mu
    
    print('Loss at true parameter: {}'.format(get_loss(true_norm_param_dict)))


.. parsed-literal::

    Loss at true parameter: 2392.751922600241


Step 2: Flatten and fold for optimization.
------------------------------------------

Note that we have written our loss, ``get_loss`` as a function of a
*dictionary of parameters*.

We can use ``paragami`` to convert such a dictionary to and from a flat,
unconstrained parameterization for optimization. (Though not show here,
it is also useful for sensitivity analysis.)

Define a ``paragami`` pattern that matches the input to ``get_loss``.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    norm_param_pattern = paragami.PatternDict()
    norm_param_pattern['sigma'] = paragami.PSDMatrixPattern(size=3)
    norm_param_pattern['mu'] = paragami.NumericArrayPattern(shape=(3, ))

“Flatten” the dictionary into an unconstrained vector.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    norm_param_freeflat = norm_param_pattern.flatten(true_norm_param_dict, free=True)
    print('The flat parameter has shape: {}'.format(norm_param_freeflat.shape))


.. parsed-literal::

    The flat parameter has shape: (9,)


Optimize using autograd.
~~~~~~~~~~~~~~~~~~~~~~~~

We can use this flat parameter to optimize the likelihood directly
without worrying about the PSD constraint on :math:`\Sigma`.

.. code:: ipython3

    print('First, wrap the loss to be a function of the flat parameter.')
    get_freeflat_loss = paragami.FlattenedFunction(
        original_fun=get_loss, patterns=norm_param_pattern, free=True)
    
    print('Now, use the flattened function to optimize with autograd.\n')
    get_freeflat_loss_grad = autograd.grad(get_freeflat_loss)
    get_freeflat_loss_hessian = autograd.hessian(get_freeflat_loss)
    
    # Initialize with zeros.
    init_param = np.zeros(norm_param_pattern.flat_length(free=True))
    mle_opt = osp.optimize.minimize(
        method='trust-ncg',
        x0=init_param,
        fun=get_freeflat_loss,
        jac=get_freeflat_loss_grad,
        hess=get_freeflat_loss_hessian,
        options={'gtol': 1e-8, 'disp': True})
    
    mle_opt = get_optimum(init_param)


.. parsed-literal::

    First, wrap the loss to be a function of the flat parameter.
    Now, use the flattened function to optimize with autograd.
    
    Warning: A bad approximation caused failure to predict improvement.
             Current function value: 2385.942776
             Iterations: 15
             Function evaluations: 17
             Gradient evaluations: 15
             Hessian evaluations: 15
    Warning: A bad approximation caused failure to predict improvement.
             Current function value: 2385.942776
             Iterations: 15
             Function evaluations: 17
             Gradient evaluations: 15
             Hessian evaluations: 15


“Fold” to inspect the result.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now “fold” the optimum back into its original shape for
inspection and further use.

.. code:: ipython3

    norm_param_opt = norm_param_pattern.fold(mle_opt.x, free=True)
    
    for param in ['sigma', 'mu']:
        print('Parmeter {}\nOptimal:\n{}\n\nTrue:\n{}\n\n'.format(
            param, norm_param_opt[param], true_norm_param_dict[param]))


.. parsed-literal::

    Parmeter sigma
    Optimal:
    [[ 1.06683522  0.07910048  0.04229475]
     [ 0.07910048  1.89297797 -0.02650233]
     [ 0.04229475 -0.02650233  2.92376984]]
    
    True:
    [[1.03745401 0.07746864 0.03950388]
     [0.07746864 2.01560186 0.05110853]
     [0.03950388 0.05110853 3.0601115 ]]
    
    
    Parmeter mu
    Optimal:
    [-0.04469438  1.03094019  1.85511868]
    
    True:
    [0 1 2]
    
    


import autograd
import autograd.numpy as np
from .function_patterns import FlattenedFunction
import scipy as osp

class HyperparameterSensitivityLinearApproximation:
    """
    Linearly approximate dependence of an optimum on a hyperparameter.

    Suppose we have an optimization problem in which the objective
    depends on a hyperparameter:

    .. math::

        \hat{\\theta} = \mathrm{argmin}_{\\theta} f(\\theta, \\lambda).

    The optimal parameter, :math:`\hat{\\theta}`, is a function of
    :math:`\\lambda` through the optimization problem.  In general, this
    dependence is complex and nonlinear.  To approximate this dependence,
    this class uses the linear approximation:

    .. math::

        \hat{\\theta}(\\lambda) \\approx \hat{\\theta}(\\lambda_0) +
            \\frac{d\hat{\\theta}}{d\\lambda}|_{\\lambda_0}
                (\\lambda - \\lambda_0).

    In terms of the arguments to this function,
    :math:`\\theta` corresponds to ``opt_par``,
    :math:`\\lambda` corresponds to ``hyper_par``,
    and :math:`f` corresponds to ``objective_fun``.

    Because ``opt_par`` and ``hyper_par`` in general are structured,
    constrained data, the linear approximation is evaluated in flattened
    space using user-specified patterns.

    Methods
    ------------
    set_base_values:
        Set the base values, :math:`\\lambda_0` and
        :math:`\\theta_0 := \hat\\theta(\\lambda_0)`, at which the linear
        approximation is evaluated.
    get_dopt_dhyper:
        Return the Jacobian matrix
        :math:`\\frac{d\hat{\\theta}}{d\\lambda}|_{\\lambda_0}` in flattened
        space.
    get_hessian_at_opt:
        Return the Hessian of the objective function in the
        flattened space.
    predict_opt_par_from_hyper_par:
        Use the linear approximation to predict
        the folded value of ``opt_par`` from a folded value of ``hyper_par``.
    """
    def __init__(
        self,
        objective_fun,
        opt_par_pattern, hyper_par_pattern,
        opt_par_folded_value, hyper_par_folded_value,
        opt_par_is_free, hyper_par_is_free,
        validate_optimum=True,
        hessian_at_opt=None,
        factorize_hessian=True,
        hyper_par_objective_fun=None):

        self._objective_fun = objective_fun
        self._opt_par_pattern = opt_par_pattern
        self._hyper_par_pattern = hyper_par_pattern
        self._opt_par_is_free = opt_par_is_free
        self._hyper_par_is_free = hyper_par_is_free

        # Define flattened versions of the objective function and their
        # autograd derivatives.
        self._obj_fun = \
            FlattenedFunction(
                original_fun=self._objective_fun,
                patterns=[self._opt_par_pattern, self._hyper_par_pattern],
                free=[self._opt_par_is_free, self._hyper_par_is_free],
                argnums=[0, 1])
        self._obj_fun_grad = autograd.grad(self._obj_fun, argnum=0)
        self._obj_fun_hessian = autograd.hessian(self._obj_fun, argnum=0)
        self._obj_fun_hvp = autograd.hessian_vector_product(
            self._obj_fun, argnum=0)

        if hyper_par_objective_fun is None:
            self._hyper_par_objective_fun = self._objective_fun
            self._hyper_obj_fun = self._obj_fun
        else:
            self._hyper_par_objective_fun = hyper_par_objective_fun
            self._hyper_obj_fun = \
                FlattenedFunction(
                    original_fun=self._hyper_par_objective_fun,
                    patterns=[self._opt_par_pattern, self._hyper_par_pattern],
                    free=[self._opt_par_is_free, self._hyper_par_is_free],
                    argnums=[0, 1])

        # TODO: is this the right default order?  Make this flexible.
        self._hyper_obj_fun_grad = autograd.grad(self._hyper_obj_fun, argnum=0)
        self._hyper_obj_cross_hess = autograd.jacobian(
            self._hyper_obj_fun_grad, argnum=1)

        self.set_base_values(
            opt_par_folded_value, hyper_par_folded_value,
            hessian_at_opt, factorize_hessian, validate=validate_optimum)

    def set_base_values(self,
                        opt_par_folded_value, hyper_par_folded_value,
                        hessian_at_opt, factorize_hessian,
                        validate=True, grad_tol=1e-8):

        # Set the values of the optimal parameters.
        self._opt0 = self._opt_par_pattern.flatten(
            opt_par_folded_value, free=self._opt_par_is_free)
        self._hyper0 = self._hyper_par_pattern.flatten(
            hyper_par_folded_value, free=self._hyper_par_is_free)

        if validate:
            # Check that the gradient of the objective is zero at the optimum.
            grad0 = self._obj_fun_grad(self._opt0, self._hyper0)
            grad0_norm = np.linalg.norm(grad0)
            if np.linalg.norm(grad0) > grad_tol:
                err_msg = \
                    'The gradient is not zero at the putatively optimal ' + \
                    'values.  ||grad|| = {} > {} = grad_tol'.format(
                        grad0_norm, grad_tol)
                raise ValueError(err_msg)

        # Set the values of the Hessian at the optimum.
        self._factorize_hessian = factorize_hessian
        if self._factorize_hessian:
            if hessian_at_opt is None:
                self._hess0 = self._obj_fun_hessian(self._opt0, self._hyper0)
            else:
                self._hess0 = hessian_at_opt
            self._hess0_chol = osp.linalg.cho_factor(self._hess0)
        else:
            if hessian_at_opt is not None:
                raise ValueError('If factorize_hessian is False, ' +
                                 'hessian_at_opt must be None.')
            self._hess0 = None
            self._hess0_chol = None

        self._cross_hess = self._hyper_obj_cross_hess(self._opt0, self._hyper0)
        self._sens_mat = -1 * osp.linalg.cho_solve(
            self._hess0_chol, self._cross_hess)

    # Methods:
    def get_dopt_dhyper(self):
        return self._sens_mat

    def get_hessian_at_opt(self):
        return self._hess0

    def predict_opt_par_from_hyper_par(self, new_hyper_par_folded_value,
                                       fold_output=True):
        hyper1 = self._hyper_par_pattern.flatten(
            new_hyper_par_folded_value, free=self._hyper_par_is_free)
        opt_par1 = self._opt0 + self._sens_mat @ (hyper1 - self._hyper0)
        if fold_output:
            return self._opt_par_pattern.fold(
                opt_par1, free=self._opt_par_is_free)
        else:
            return opt_par1

# TODO:
# Preconditioned functions

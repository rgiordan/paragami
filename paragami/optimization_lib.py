import autograd
import autograd.numpy as np
import copy
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
        hyper_par_objective_fun=None,
        grad_tol=1e-8):
        """
        Parameters
        --------------
        objective_fun: Callable function
            A callable function, optimized by ``opt_par`` at a particular value
            of ``hyper_par``.  The function must be of the form
            ``f(folded opt_par, folded hyper_par)``.
        opt_par_pattern:
            A pattern for ``opt_par``, the optimal parameter.
        opt_par_pattern:
            A pattern for ``hyper_par``, the hyperparameter.
        opt_par_folded_value:
            The folded value of ``opt_par`` at which ``objective_fun`` is
            optimized for the given value of ``hyper_par``.
        hyper_par_folded_value:
            The folded of ``hyper_par_folded_value`` at which ``opt_par``
            optimizes ``objective_fun``.
        opt_par_is_free: Boolean
            Whether to use the free parameterization for ``opt_par``` when
            linearzing.
        hyper_par_is_free: Boolean
            Whether to use the free parameterization for ``hyper_par``` when
            linearzing.
        validate_optimum: Boolean
            When setting the values of ``opt_par`` and ``hyper_par``, check
            that ``opt_par`` is, in fact, a critical point of
            ``objective_fun``.
        hessian_at_opt: Numeric matrix (optional)
            The Hessian of ``objective_fun`` at the optimum.  If not specified,
            it is calculated using automatic differentiation.
        factorize_hessian: Boolean
            If ``True``, solve the required linear system using a Cholesky
            factorization.  If ``False``, use the conjugate gradient algorithm
            to avoid forming or inverting the Hessian.
        hyper_par_objective_fun: Callable function
            A callable function of the form
            ``f(folded opt_par, folded hyper_par)`` containing the part of
            ``objective_fun`` that depends on both ``opt_par`` and
            ``hyper_par``.  If not specified, ``objective_fun`` is used.
        grad_tol: Float
            The tolerance used to check that the gradient is approximately
            zero at the optimum.
        """

        self._objective_fun = objective_fun
        self._opt_par_pattern = opt_par_pattern
        self._hyper_par_pattern = hyper_par_pattern
        self._opt_par_is_free = opt_par_is_free
        self._hyper_par_is_free = hyper_par_is_free
        self._grad_tol = grad_tol

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
                        validate=True, grad_tol=None):
        if grad_tol is None:
            grad_tol = self._grad_tol

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
        """
        Predict ``opt_par`` using the linear approximation.

        Parameters
        ------------
        new_hyper_par_folded_value:
            The folded value of ``hyper_par`` at which to approximate
            ``opt_par``.
        fold_output: Boolean
            Whether to return ``opt_par`` as a folded value.  If ``False``,
            returns the flattened value according to ``opt_par_pattern``
            and ``opt_par_is_free``.
        """

        if not self._factorize_hessian:
            raise NotImplementedError(
                'CG is not yet implemented for predict_opt_par_from_hyper_par')

        hyper1 = self._hyper_par_pattern.flatten(
            new_hyper_par_folded_value, free=self._hyper_par_is_free)
        opt_par1 = self._opt0 + self._sens_mat @ (hyper1 - self._hyper0)
        if fold_output:
            return self._opt_par_pattern.fold(
                opt_par1, free=self._opt_par_is_free)
        else:
            return opt_par1



###############################
# Preconditioned objectives.  #
###############################


def _get_sym_matrix_inv_sqrt(mat, ev_min=None, ev_max=None):
    """
    Get the inverse square root of a symmetric matrix with thresholds for the
    eigenvalues.

    This is particularly useful for calculating preconditioners.
    """
    mat = np.atleast_2d(mat)

    # Symmetrize for numerical stability.
    mat_sym = 0.5 * (mat + mat.T)
    eig_val, eig_vec = np.linalg.eigh(mat_sym)

    if not ev_min is None:
        if not np.isreal(ev_min):
            raise ValueError('ev_min must be real-valued.')
        eig_val[np.real(eig_val) <= ev_min] = ev_min
    if not ev_max is None:
        if not np.isreal(ev_max):
            raise ValueError('ev_max must be real-valued.')
        eig_val[np.real(eig_val) >= ev_max] = ev_max

    mat_corrected = np.matmul(eig_vec,
                               np.matmul(np.diag(eig_val), eig_vec.T))
    mat_sqrt = \
        np.matmul(eig_vec,
                  np.matmul(np.diag(np.sqrt(eig_val)), eig_vec.T))

    mat_inv_sqrt = \
        np.matmul(eig_vec,
                  np.matmul(np.diag(1 / np.sqrt(eig_val)), eig_vec.T))

    return np.array(mat_inv_sqrt), \
           np.array(mat_sqrt), \
           np.array(mat_corrected)


class PreconditionedFunction():
    """
    Get a function whose input has been preconditioned.

    Throughout, the subscript ``_c`` will denote quantiites or
    funcitons in the preconditioned space.  For example, ``x`` will
    refer to a variable in the original space and ``x_c`` to the same
    variable after preconditioning.

    Preconditioning means transforming :math:`x \\rightarrow x_c = A^{-1} x`,
    where the matrix :math:`A` is the "preconditioner".  If :math:`f` operates
    on :math:`x`, then the preconditioned function operates on :math:`x_c` and
    is defined by :math:`f_c(x_c) := f(A x_c) = f(x)`. Gradients of the
    preconditioned function are defined with respect to its argument in the
    preconditioned space, e.g., :math:`f'_c = \\frac{df_c}{dx_c}`.

    A typical value of the preconditioner is an inverse square root of the
    Hessian of :math:`f`, because then the Hessian of :math:`f_c` is
    the identity when the gradient is zero.  This can help speed up the
    convergence of optimization algorithms.

    Methods
    ----------
    set_preconditioner:
        Set the preconditioner to a specified value.
    set_preconditioner_with_hessian:
        Set the preconditioner based on the Hessian of the objective
        at a point in the orginal domain.
    get_preconditioner:
        Return a copy of the current preconditioner.
    get_preconditioner_inv:
        Return a copy of the current inverse preconditioner.
    precondition:
        Convert from the original domain to the preconditioned domain.
    unprecondition:
        Convert from the preconditioned domain to the original domain.
    """
    def __init__(self, original_fun,
                 preconditioner=None,
                 preconditioner_inv=None):
        """
        Parameters
        -------------
        original_fun:
            callable function of a single argument
        preconditioner:
            The initial preconditioner.
        preconditioner_inv:
            The inverse of the initial preconditioner.
        """
        self._original_fun = original_fun
        self._original_fun_hessian = autograd.hessian(self._original_fun)
        if (preconditioner is None) and (preconditioner_inv is not None):
            raise ValueError(
                'If you specify preconditioner_inv, you must' +
                'also specify preconditioner. ')
        if preconditioner is not None:
            self.set_preconditioner(preconditioner, preconditioner_inv)
        else:
            self._preconditioner = None
            self._preconditioner_inv = None

    def get_preconditioner(self):
        return copy.copy(self._preconditioner)

    def get_preconditioner_inv(self):
        return copy.copy(self._preconditioner_inv)

    def set_preconditioner(self, preconditioner, preconditioner_inv=None):
        self._preconditioner = preconditioner
        if preconditioner_inv is None:
            self._preconditioner_inv = np.linalg.inv(self._preconditioner)
        else:
            self._preconditioner_inv = preconditioner_inv

    def set_preconditioner_with_hessian(self, x=None, hessian=None,
                                        ev_min=None, ev_max=None):
        """
        Set the precoditioner to the inverse square root of the Hessian of
        the original objective (or an approximation thereof).

        Parameters
        ---------------
        x: Numeric vector
            The point at which to evaluate the Hessian of ``original_fun``.
            If x is specified, the Hessian is evaluated with automatic
            differentiation.
            Specify either x or hessian but not both.
        hessian: Numeric matrix
            The hessian of ``original_fun`` or an approximation of it.
            Specify either x or hessian but not both.
        ev_min: float
            If not None, set eigenvaluse of ``hessian`` that are less than
            ``ev_min`` to ``ev_min`` before taking the square root.
        ev_maxs: float
            If not None, set eigenvaluse of ``hessian`` that are greater than
            ``ev_max`` to ``ev_max`` before taking the square root.

        Returns
        ------------
        Sets the precoditioner for the class and returns the Hessian with
        the eigenvalues thresholded by ``ev_min`` and ``ev_max``.
        """
        if x is not None and hessian is not None:
            raise ValueError('You must specify x or hessian but not both.')
        if x is None and hessian is None:
            raise ValueError('You must specify either x or hessian.')
        if hessian is None:
            # We now know x is not None.
            hessian = self._original_fun_hessian(x)

        hess_inv_sqrt, hess_sqrt, hess_corrected = \
            _get_sym_matrix_inv_sqrt(hessian, ev_min, ev_max)
        self.set_preconditioner(hess_inv_sqrt, hess_sqrt)

        return hess_corrected

    def precondition(self, x):
        """
        Multiply by the inverse of the preconditioner to convert
        :math:`x` in the original domain to :math:`x_c` in the preconditioned
        domain.

        This function is provided for convenience, but it is more numerically
        stable to use np.linalg.solve(preconditioner, x).
        """
        # On one hand, this is a numerically instable way to solve a linear
        # system.  On the other hand, the inverse is readily available from
        # the eigenvalue decomposition and the Cholesky factorization
        # is not AFAIK.
        if self._preconditioner_inv is None:
            raise ValueError('You must set the preconditioner.')
        return self._preconditioner_inv @ x

    def unprecondition(self, x_c):
        """
        Multiply by the preconditioner to convert
        :math:`x_c` in the preconditioned domain to :math:`x` in the
        original domain.
        """
        if self._preconditioner is None:
            raise ValueError('You must set the preconditioner.')
        return self._preconditioner @ x_c

    def __call__(self, x_c):
        """
        Evaluate the preconditioned function at a point in the preconditioned
        domain.
        """
        return self._original_fun(self.unprecondition(x_c))

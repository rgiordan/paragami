import autograd
import autograd.numpy as np
import copy
import scipy as osp

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

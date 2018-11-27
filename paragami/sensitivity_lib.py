import autograd
import autograd.numpy as np
import copy
from .function_patterns import FlattenedFunction


#######################################
# Higher-order Taylor series         #
######################################

import autograd
import autograd.numpy as np

import scipy as sp

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.SparseObjectives as obj_lib

from copy import deepcopy

import math

def set_par(par, val, is_free):
    if is_free:
        par.set_free(val)
    else:
        par.set_vector(val)


def _append_jvp(fun, num_base_args=1, argnum=0):
    """
    Append a jacobian vector product to a function.

    This function is designed to be used recursively to calculate
    higher-order Jacobian-vector products.

    Parameters
    --------------
    fun: Callable function
        The function to be differentiated.
    num_base_args: integer
        The number of inputs to the base function, i.e.,
        to the function before any differentiation.
     argnum: inteeger
        Which argument should be differentiated with respect to.
        Must be between 0 and num_base_args - 1.

    Returns
    ------------
    Denote the base args x1, ..., xB, where B == num_base_args.
    Let argnum = k.  Then _append_jvp returns a function,
    fun_jvp(x1, ..., xB, ..., v) =
    \sum_i (dfun_dx_{ki}) v_i | (x1, ..., xB).
    That is, it returns the Jacobian vector product where the Jacobian
    is taken with respect to xk, and the vector product is with the
    final argument.
    """
    assert argnum < num_base_args

    fun_jvp = autograd.make_jvp(fun, argnum=argnum)
    def obj_jvp_wrapper(*argv):
        # These are the base arguments -- the points at which the
        # Jacobians are evaluated.
        base_args = argv[0:num_base_args]

        # The rest of the arguments are the vectors, with which inner
        # products are taken in the order they were passed to
        # _append_jvp.
        vec_args = argv[num_base_args:]

        if (len(vec_args) > 1):
            # Then this is being applied to an existing Jacobian
            # vector product.  The last will be the new vector, and
            # we need to evaluate the function at the previous vectors.
            # The new jvp will be appended to the end of the existing
            # list.
            old_vec_args = vec_args[:-1]
            return fun_jvp(*base_args, *old_vec_args)(vec_args[-1])[1]
        else:
            return fun_jvp(*base_args)(*vec_args)[1]

    return obj_jvp_wrapper


class DerivativeTerm:
    """
    A single term in a Taylor expansion of a two-parameter objective with
    methods for computing its derivatives.

    The nomenclature assumes that
    we are calculating derivatives of g(eta, eps) at (eta0, eps0).  This
    can be used to calculate d^k\hat\eta / d\eps^k | (eta0, eps0) where
    \hat\eta: g(\hat\eta, \eps) = 0.

    Attributes
    -----------------
    See the arguments to ```__init___```.

    Methods
    ------------
    evaluate:
        Get the value of the current derivative term.
    differentiate:
        Get a list of derivatives terms resulting from differentiating this
        term.
    check_similarity:
        Return a boolean indicating whether this term is equivalent to another
        term in the order of its derivative.
    combine_with:
        Return the sum of this term and another term.
    """
    def __init__(self, eps_order, eta_orders, prefactor,
                 eval_eta_derivs, eval_g_derivs):
        """
        Parameters
        -------------
        eps_order:
            The total number of epsilon derivatives of g.
        eta_orders:
            A vector of length order - 1.  Entry i contains the number
            of terms d\eta^{i + 1} / d\epsilon^{i + 1}.
        prefactor:
            The constant multiple in front of this term.
        eval_eta_derivs:
            A vector of functions to evaluate d\eta^i / d\epsilon^i.
            The functions should take arguments (eta0, eps0, deps) and the
            i^{th} entry should evaluate
            d\eta^i / d\epsilon^i (deps^i) |_{eta0, eps0}.
        eval_g_derivs:
            A list of lists of g jacobian vector product functions.
            The array should be such that
            eval_g_derivs[i][j](eta0, eps0, v1 ... vi, w1 ... wj)
            evaluates d^{i + j} G / (deta^i)(deps^j)(v1 ... vi)(w1 ... wj).
        """
        # Base properties.
        self.eps_order = eps_order
        self.eta_orders = eta_orders
        self.prefactor = prefactor
        self.eval_eta_derivs = eval_eta_derivs
        self.eval_g_derivs = eval_g_derivs

        # Derived quantities.

        # The order is the total number of epsilon derivatives.
        self.order = int(
            self.eps_order + \
            np.sum(self.eta_orders * np.arange(1, len(self.eta_orders) + 1)))

        # The derivative of g needed for this particular term.
        self.eval_g_deriv = \
            eval_g_derivs[np.sum(eta_orders)][self.eps_order]

        # Sanity checks.
        # The rules of differentiation require that these assertions be true
        # -- that is, if terms are generated using the differentiate()
        # method from other well-defined terms, these assertions should always
        # be sastisfied.
        assert isinstance(self.eps_order, int)
        assert len(self.eta_orders) == self.order
        assert self.eps_order >= 0 # Redundant
        for eta_order in self.eta_orders:
            assert eta_order >= 0
            assert isinstance(eta_order, int)
        assert len(self.eval_eta_derivs) >= self.order - 1
        assert len(eval_g_derivs) > len(self.eta_orders)
        for eta_deriv_list in eval_g_derivs:
            assert len(eta_deriv_list) > self.eps_order

    def __str__(self):
        return 'Order: {}\t{} * eta{} * eps[{}]'.format(
            self.order, self.prefactor, self.eta_orders, self.eps_order)

    def evaluate(self, eta0, eps0, deps):
        # First eta arguments, then epsilons.
        vec_args = []

        for i in range(len(self.eta_orders)):
            eta_order = self.eta_orders[i]
            if eta_order > 0:
                vec = self.eval_eta_derivs[i](eta0, eps0, deps)
                for j in range(eta_order):
                    vec_args.append(vec)

        for i in range(self.eps_order):
            vec_args.append(deps)

        return self.prefactor * self.eval_g_deriv(eta0, eps0, *vec_args)

    def differentiate(self, eval_next_eta_deriv):
        derivative_terms = []
        new_eval_eta_derivs = deepcopy(self.eval_eta_derivs)
        new_eval_eta_derivs.append(eval_next_eta_deriv)

        old_eta_orders = deepcopy(self.eta_orders)
        old_eta_orders.append(0)

        # dG / deps.
        #print('dg/deps')
        derivative_terms.append(
            DerivativeTerm(
                eps_order=self.eps_order + 1,
                eta_orders=deepcopy(old_eta_orders),
                prefactor=self.prefactor,
                eval_eta_derivs=new_eval_eta_derivs,
                eval_g_derivs=self.eval_g_derivs))

        # dG / deta.
        #print('dg/deta')
        new_eta_orders = deepcopy(old_eta_orders)
        new_eta_orders[0] = new_eta_orders[0] + 1
        derivative_terms.append(
            DerivativeTerm(
                eps_order=self.eps_order,
                eta_orders=new_eta_orders,
                prefactor=self.prefactor,
                eval_eta_derivs=new_eval_eta_derivs,
                eval_g_derivs=self.eval_g_derivs))

        # Derivatives of each d^{i}eta / deps^i term.
        for i in range(len(self.eta_orders)):
            #print('i: ', i)
            eta_order = self.eta_orders[i]
            if eta_order > 0:
                new_eta_orders = deepcopy(old_eta_orders)
                new_eta_orders[i] = new_eta_orders[i] - 1
                new_eta_orders[i + 1] = new_eta_orders[i + 1] + 1
                derivative_terms.append(
                    DerivativeTerm(
                        eps_order=self.eps_order,
                        eta_orders=new_eta_orders,
                        prefactor=self.prefactor * eta_order,
                        eval_eta_derivs=new_eval_eta_derivs,
                        eval_g_derivs=self.eval_g_derivs))

        return derivative_terms

    # Return whether another term matches this one in the pattern of derivatives.
    def check_similarity(self, term):
        return \
            (self.eps_order == term.eps_order) & \
            (self.eta_orders == term.eta_orders)

    # Assert that another term has the same pattern of derivatives and
    # return a new term that combines the two.
    def combine_with(self, term):
        assert self.check_similarity(term)
        return DerivativeTerm(
            eps_order=self.eps_order,
            eta_orders=self.eta_orders,
            prefactor=self.prefactor + term.prefactor,
            eval_eta_derivs=self.eval_eta_derivs,
            eval_g_derivs=self.eval_g_derivs)


def _generate_two_term_derivative_array(fun, order):
    """
    Generate an array of JVPs of the two arguments of the target function fun.

    Parameters
    -------------
    fun: callable function
        The function to be differentiated.  The first two arguments
        should be vectors for differentiation, i.e., fun should have signature
        fun(x1, x2, ...) and return a numeric value.
     order: integer
        The maximum order of the derivative to be generated.

    Returns
    ------------
    An array of functions where element eval_fun_derivs[i][j] is a function
    eval_fun_derivs[i][j](x1, x2, ..., v1, ... vi, w1, ..., wj)) =
    d^{i + j}fun / (dx1^i dx2^j) v1 ... vi w1 ... wj.
    """
    eval_fun_derivs = [[ fun ]]
    for x1_ind in range(order):
        if x1_ind > 0:
            # Append one x1 derivative.
            next_deriv = _append_jvp(
                eval_fun_derivs[x1_ind - 1][0], num_base_args=2, argnum=0)
            eval_fun_derivs.append([ next_deriv ])
        for x2_ind in range(order):
            # Append one x2 derivative.
            next_deriv = _append_jvp(
                eval_fun_derivs[x1_ind][x2_ind], num_base_args=2, argnum=1)
            eval_fun_derivs[x1_ind].append(next_deriv)
    return eval_fun_derivs



def _consolidate_terms(dterms):
    """
    Combine like derivative terms.

    Arguments
    -----------
    dterms:
        A list of DerivativeTerms.

    Returns
    ------------
    A new list of derivative terms that evaluate equivalently where
    terms with the same derivative signature have been combined.
    """
    unmatched_indices = [ ind for ind in range(len(dterms)) ]
    consolidated_dterms = []
    while len(unmatched_indices) > 0:
        match_term = dterms[unmatched_indices.pop(0)]
        for ind in unmatched_indices:
            if (match_term.eta_orders == dterms[ind].eta_orders):
                match_term = match_term.combine_with(dterms[ind])
                unmatched_indices.remove(ind)
        consolidated_dterms.append(match_term)

    return consolidated_dterms

def evaluate_terms(dterms, eta0, eps0, deps, include_highest_eta_order=True):
    """
    Evaluate a list of derivative terms.

    Parameters
    ---------------
    dterms:
        A list of derivative terms.
    eta0:
        The value of the first argument at which the derivative is evaluated.
    eps0:
        The value of the second argument at which the derivative is evaluated.
    deps: numpy array
        The change in epsilon by which to multiply the Jacobians.
    include_highest_eta_order: boolean
        If true, include the term with
        d^k eta / deps^k, where k == order.  The main use of these
        DerivativeTerms at the time of writing is precisely to evaluate this
        term using the other terms, and this can be accomplished by setting
        include_highest_eta_order to False.

    Returns
    ---------------
        The sum of the evaluated DerivativeTerms.
    """
    vec = None
    for term in dterms:
        if include_highest_eta_order or (term.eta_orders[-1] == 0):
            if vec is None:
                vec = term.evaluate(eta0, eps0, deps)
            else:
                vec += term.evaluate(eta0, eps0, deps)
    return vec


# Get the terms to start a Taylor expansion.
def _get_taylor_base_terms(eval_g_derivs):
    dterms1 = [ \
        DerivativeTerm(
            eps_order=1,
            eta_orders=[0],
            prefactor=1.0,
            eval_eta_derivs=[],
            eval_g_derivs=eval_g_derivs),
        DerivativeTerm(
            eps_order=0,
            eta_orders=[1],
            prefactor=1.0,
            eval_eta_derivs=[],
            eval_g_derivs=eval_g_derivs) ]
    return dterms1


# Given a collection of dterms (formed either with _get_taylor_base_terms
# or derivatives), evaluate the implied dketa_depsk.
#
# Args:
#   - hess0: The Hessian of the objective wrt the first argument.
#   - dterms: An array of DerivativeTerms.
#   - eta0: The value of the first argument.
#   - eps0: The value of the second argument.
#   - deps: The change in epsilon by which to multiply the Jacobians.
def evaluate_dketa_depsk(hess0, dterms, eta0, eps0, deps):
    vec = evaluate_terms(
        dterms, eta0, eps0, deps, include_highest_eta_order=False)
    assert vec is not None
    return -1 * np.linalg.solve(hess0, vec)


# Calculate the derivative of an array of DerivativeTerms.
#
# Args:
#   - hess0: The Hessian of the objective wrt the first argument.
#   - dterms: An array of DerivativeTerms.
#
# Returns:
#   An array of the derivatives of dterms with respect to the second argument.
def differentiate_terms(hess0, dterms):
    def eval_next_eta_deriv(eta, eps, deps):
        return evaluate_dketa_depsk(hess0, dterms, eta, eps, deps)

    dterms_derivs = []
    for term in dterms:
        dterms_derivs += term.differentiate(eval_next_eta_deriv)
    return _consolidate_terms(dterms_derivs)
    return dterms_derivs



class ParametricSensitivityTaylorExpansionForwardDiff(object):
    """
    This is a class for computing the Taylor series of
    eta(eps) = argmax_eta objective(eta, eps) using forward-mode automatic
    differentation.

    Methods:
      - evaluate_dkinput_dhyperk.
          Args:
              - dhyper: The difference hyper_val1 - hyper_val0 by which the
              derivatives are multiplied.
              - k: The order of the derivative to return.
          Returns:
              dkinput_dhyperk * (hyper_val1 - hyper_val0).
      - evaluate_taylor_series
          Args:
              - dhyper: The difference hyper_val1 - hyper_val0 by which the
              derivatives are multiplied.
              - add_offset: Whether the Taylor expansion includes the
              offset input_val0.
              - max_order: The maximum order of the Taylor expansion to use.
              Higher orders will be slower to evaluate.  Defaults to the
              highest order possible, i.e. the order specified at initialization.
          Returns:
              The value of the Taylor series expansion of \hat\eta(\eps) at
              hyper_val0 + dhyper.
    """
    def __init__(
        self, objective_functor,
        input_val0, hyper_val0, order,
        input_is_free=True, hyper_is_free=False,
        hess0=None, hyper_par_objective_functor=None):
        """
        Parameters
        ------------------
        objective_function: callable function.
            The optimization objective as a function of two arguments
            (eta, eps), where eta is the parameter that is optimized and
            eps is a hyperparameter.
        input_val0: numpy array
            The value of input_par at the optimum.
        hyper_val0: numpy array
            The value of hyper_par at which input_val0 was found.
        order: positive integer
            The maximum order of the Taylor series to be calculated.
        hess0: numpy array
            Optional.  The Hessian of the objective at (input_val0, hyper_val0).
            If not specified it is calculated at initialization.
         hyper_par_objective_funcion:
            Optional.  A function containing the dependence
            of objective_functor on the hyperparameter.  Sometimes only a small,
            easily calculated part of the objective depends on the
            hyperparameter, and by specifying hyper_par_objective_functor the
            necessary calculations can be more efficient.  If unset,
            ```objective_function``` is used.
        """

        self.objective_functor = objective_functor
        self.input_par = input_par
        self.hyper_par = hyper_par
        self.input_is_free = input_is_free
        self.hyper_is_free = hyper_is_free

        if hyper_par_objective_functor is None:
            self.hyper_par_objective_functor = objective_functor
        else:
            self.hyper_par_objective_functor = hyper_par_objective_functor

        self.objective = obj_lib.Objective(
            self.input_par, self.objective_functor)
        self.joint_objective = obj_lib.TwoParameterObjective(
            self.input_par, self.hyper_par, self.hyper_par_objective_functor)

        self.set_base_values(input_val0, hyper_val0)
        self.set_order(order)

    def _set_par_to_base_values(self):
        set_par(self.input_par, self.input_val0, self.input_is_free)
        set_par(self.hyper_par, self.hyper_val0, self.hyper_is_free)

    def _cache_and_eval(self, diff_fun, *argv, **argk):
        result = diff_fun(*argv, **argk)
        self.set_par_to_base_values()
        return result

    def _set_base_values(self, input_val0, hyper_val0, hess0=None):
        self.input_val0 = deepcopy(input_val0)
        self.hyper_val0 = deepcopy(hyper_val0)
        self.set_par_to_base_values()

        if hess0 is None:
            self.hess0 = self.objective.fun_free_hessian(self.input_val0)
        else:
            self.hess0 = hess0
        self.hess0_chol = sp.linalg.cho_factor(self.hess0)

    # In order to calculate derivatives d^kinput_dhyper^k, we will be Taylor
    # expanding the gradient of the objective.
    def objective_gradient(self, input_val, hyper_val, *argv, **argk):
        return self.joint_objective.fun_grad1(
            input_val, hyper_val, self.input_is_free, self.hyper_is_free,
            *argv, **argk)

    # Get a function returning the next derivative from the Taylor terms dterms.
    def get_dkinput_dhyperk_from_terms(self, dterms):
        def dkinput_dhyperk(input_val, hyper_val, dhyper, tolerance=1e-8):
            if tolerance is not None:
                # Make sure you're evaluating sensitivity at the base parameters.
                assert np.max(np.abs(input_val - self.input_val0)) <= tolerance
                assert np.max(np.abs(hyper_val - self.hyper_val0)) <= tolerance
            return evaluate_dketa_depsk(
                self.hess0, dterms, self.input_val0, self.hyper_val0, dhyper)
        return dkinput_dhyperk

    def differentiate_terms(self, dterms, eval_next_eta_deriv):
        dterms_derivs = []
        for term in dterms:
            dterms_derivs += term.differentiate(eval_next_eta_deriv)
        return _consolidate_terms(dterms_derivs)

    def set_order(self, order):
        self.order = order

        # You need one more gradient derivative than the order of the Taylor
        # approximation.
        self.eval_g_derivs = _generate_two_term_derivative_array(
            self.objective_gradient, order=self.order + 1)

        self.taylor_terms_list = [ _get_taylor_base_terms(self.eval_g_derivs) ]
        self.dkinput_dhyperk_list = []
        for k in range(self.order - 1):
            next_dkinput_dhyperk = self.get_dkinput_dhyperk_from_terms(
                self.taylor_terms_list[k])
            next_taylor_terms = self.differentiate_terms(
                self.taylor_terms_list[k],
                next_dkinput_dhyperk)
            self.dkinput_dhyperk_list.append(next_dkinput_dhyperk)
            self.taylor_terms_list.append(next_taylor_terms)

        self.dkinput_dhyperk_list.append(
            self.get_dkinput_dhyperk_from_terms(
                self.taylor_terms_list[self.order - 1]))

    def evaluate_dkinput_dhyperk(self, dhyper, k, *argv, **argk):
        if k <= 0:
            raise ValueError('k must be at least one.')
        if k > self.order:
            raise ValueError(
                'k must be no greater than the declared order={}'.format(
                    self.order))
        deriv_fun = self.dkinput_dhyperk_list[k - 1]
        return self.cache_and_eval(
            deriv_fun, self.input_val0, self.hyper_val0, dhyper, *argv, **argk)

    def evaluate_taylor_series(
        self, dhyper, *argv, add_offset=True, max_order=None, **argk):
        if max_order is None:
            max_order = self.order
        if max_order <= 0:
            raise ValueError('max_order must be greater than zero.')
        if max_order > self.order:
            raise ValueError(
                'max_order must be no greater than the declared order={}'.format(
                    self.order))

        dinput = self.evaluate_dkinput_dhyperk(dhyper, 1, *argv,  **argk)
        for k in range(2, max_order + 1):
            dinput += self.evaluate_dkinput_dhyperk(dhyper, k) / \
                float(math.factorial(k))

        if add_offset:
            return dinput + self.input_val0
        else:
            return dinput

    def print_terms(self, k=None):
        if k is not None and k > self.order:
            raise ValueError(
                'k must be no greater than order={}'.format(self.order))
        for order in range(self.order):
            if k is None or order == (k - 1):
                print('\nTerms for order {}:'.format(order + 1))
                for term in self.taylor_terms_list[order]:
                    print(term)



# This is a class for computing a linear approximation to
# input_par(hyper_par) = argmax_input_par objective(input_par, hyper_par).
# This approximation uses reverse mode automatic differentiation.
#
# Args:
#   - objective_functor: A functor that evaluates an optimization objective
#   with respect to input_par at hyperparameter hyper_par.
#   - input_par: The input Parameter for the optimization problem:
#   - hyper_par: The hyperparameter for the optimization problem.
#   - input_val0: The value of input_par at the optimum.
#   - hyper_val0: The value of hyper_par at which input_val0 was found.
#   - input_is_free: Whether or not input_val0 is the free (vs vector) value
#   for input_par.  Defaults to free.
#   - hyper_is_free: Whether or not hyper_val0 is the free (vs vector) value
#   for hyper_par.  Defaults to not free (i.e. to vector).
#   - hess0: Optional, the Hessian of the objective at (input_val0, hyper_val0).
#   If not specified it is calculated at initialization.
#   - hyper_par_objective_functor: Optional, a functor containing the dependence
#   of objective_functor on the hyperparameter.  Sometimes only a small,
#   easily calculated part of the objective depends on the hyperparameter,
#   and by specifying hyper_par_objective_functor the necessary calculations
#   can be more efficient.  If unset, objective_functor is used.
#
# Methods:
#  - get_dinput_dhyper:
#   Args: None.
#   Returns:
#       The Jacobian matrix d(input_par) / d(hyper_par) evaluated at
#       input_val0..
#
#  - predict_input_par_from_hyperparameters:
#   Args:
#       new_hyper_par_value: A new value of the hyperparameters (either
#           as a free or constrained vector according to hyper_is_free).
#   Returns:
#       A linear approximation to input_par(hyper_par) evaluated at input_val0.
class ParametricSensitivityLinearApproximation(object):
    def __init__(
        self, objective_functor,
        input_par, hyper_par,
        input_val0, hyper_val0,
        input_is_free=True, hyper_is_free=False,
        hess0=None, hyper_par_objective_functor=None):

        self.objective_functor = objective_functor
        self.input_par = input_par
        self.hyper_par = hyper_par
        self.input_is_free = input_is_free
        self.hyper_is_free = hyper_is_free

        if hyper_par_objective_functor is None:
            self.hyper_par_objective_functor = objective_functor
        else:
            self.hyper_par_objective_functor = hyper_par_objective_functor

        self.objective = obj_lib.Objective(
            self.input_par, self.objective_functor)
        self.joint_objective = obj_lib.TwoParameterObjective(
            self.input_par, self.hyper_par, self.hyper_par_objective_functor)

        self.set_base_values(input_val0, hyper_val0)

    def set_par_to_base_values(self):
        set_par(self.input_par, self.input_val0, self.input_is_free)
        set_par(self.hyper_par, self.hyper_val0, self.hyper_is_free)

    def set_base_values(self, input_val0, hyper_val0, hess0=None):
        self.input_val0 = deepcopy(input_val0)
        self.hyper_val0 = deepcopy(hyper_val0)
        self.set_par_to_base_values()

        if hess0 is None:
            self.hess0 = self.objective.fun_free_hessian(self.input_val0)
        else:
            self.hess0 = hess0
        self.hess0_chol = sp.linalg.cho_factor(self.hess0)

        self.hyper_par_cross_hessian0 = \
            self.joint_objective.fun_hessian_free1_vector2(
                self.input_val0, self.hyper_val0)

        self.hyper_par_sensitivity = \
            -1 * sp.linalg.cho_solve(
                self.hess0_chol, self.hyper_par_cross_hessian0)

    # Methods:
    def get_dinput_dhyper(self):
        return self.hyper_par_sensitivity

    def predict_input_par_from_hyperparameters(self, new_hyper_par_value):
        hyper_par_diff = new_hyper_par_value - self.hyper_val0
        return \
            self.input_val0 + \
            self.hyper_par_sensitivity @ hyper_par_diff

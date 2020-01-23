from typing import List, Optional, Union

import numpy as np
import wandb

import wandb_util
from util import EdgeVector, IS_EXPERIMENT



class ModuloSimplexPivoting:
    #: Whether the algorithm should run verbose (i.e. print out its status).
    _verbose: bool
    #: Stores the current iteration (used for logging to wandb).
    _iteration: int

    #: The global time period (:math:`T`).
    _time_period: int

    #: The weights for each edge.
    _weights: EdgeVector

    #: The current basic variables.
    _basic_variables: List[str]
    #: The current non-basic variables.
    _non_basic_variables: List[str]

    #: The coefficition matrix of the simplex tableau.
    _gamma: np.ndarray
    #: The right-hand-side of the simplex tableau.
    _rhs: np.ndarray


    def __init__(self, time_period: int, basic_variables: list, non_basic_variables: list, gamma: np.ndarray, rhs: np.ndarray,
                 weights: EdgeVector, verbose: bool = False):
        """
        Initializes the network modulo simplex algorithm.

        Note that the algorithm changes the contents of the following parameters:
            * ``basic_variables``
            * ``non_basic_variables``
            * ``gamma``
            * ``rhs``

        :param int time_period: Time period, namely :math:`T`.
        :param list basic_variables: Basic variables (i.e. the co-tree arcs) as ordered in ``gamma``.
        :param list non_basic_variables: Non-basic variables (i.e. the tree arcs) as ordered in ``gamma``.
        :param np.ndarray gamma: Separated edge cycle matrix.
        :param np.ndarray rhs: Right hand side of the simplex tableau, namely :math:`b`.
        :param EdgeVector weights: Weights of the arcs. Must contain all tree and co-tree arc weights.
        :param bool verbose: Whether to print the status (True) or not (False).
        """

        self._verbose = verbose

        self._time_period = time_period
        self._basic_variables = basic_variables
        self._non_basic_variables = non_basic_variables
        self._gamma = gamma
        self._rhs = rhs
        self._weights = weights

        self._iteration = 0


    def perform_pivoting(self) -> float:
        """
        Performs pivoting steps until no further improvement can be done (leading to a local minimum).

        :param bool log: Whether to log the process to stdout or not.
        """

        if self._verbose:
            print('\nInitial simplex tableau and cost:')
            self._print_simplex_tableau()
            print('Starting modulo simplex pivoting.')
        self._iteration = 0
        while True:
            wandb_util.uber_iteration += 1

            pivoting_cost_change = self._perform_pivoting_step()
            cost = self._calculate_cost()

            if IS_EXPERIMENT:
                wandb.log({ 'cost': cost, 'cost_change': pivoting_cost_change }, step = wandb_util.uber_iteration)

            if pivoting_cost_change:
                self._iteration += 1

                if self._verbose:
                    print('\nSimplex tableau after %d iterations (last cost change: %.2f):' % (
                            self._iteration, pivoting_cost_change))
                    self._print_simplex_tableau(cost)
            else:
                break

        if self._verbose:
            print('\nPivoting finished after %d iterations! Resulting tableau and cost:' % self._iteration)
            self._print_simplex_tableau(cost)

        return self._calculate_cost()


    def _perform_pivoting_step(self) -> Union[float, bool]:
        """
        Performs a single modulo simplex pivoting step, i.e.:
            1. Search a pivoting element.
            2. Perform the actual pivoting and update all relevant fields.

        :return: If the pivoting has been performed, the cost change (``float``). If it was not possible, ``False`` (``bool``).
        :rtype: float or bool
        """

        # Compute the optimal pivot element.
        cost_change_matrix = np.zeros((len(self._non_basic_variables), len(self._basic_variables)))
        for i, non_basic_var in enumerate(self._non_basic_variables):
            for j, basic_var in enumerate(self._basic_variables):
                cost_change_matrix[i, j] = self._calculate_cost_difference(i, j)
        (pivot_i, pivot_j) = np.unravel_index(np.argmin(cost_change_matrix, axis = None), cost_change_matrix.shape)
        pivot_cost_change = cost_change_matrix[pivot_i, pivot_j]
        if pivot_cost_change >= 0:
            # The minimal cost change is not negative, that is not good --> no improvement possible.
            return False

        pivot_non_basic_variable = self._non_basic_variables[pivot_i]
        pivot_basic_variable = self._basic_variables[pivot_j]
        pivot_gamma = self._gamma[pivot_i, pivot_j]
        pivot_rhs = self._rhs[pivot_i]

        gamma_new = self._gamma.copy()
        for i, non_basic_var in enumerate(self._non_basic_variables):
            for j, basic_var in enumerate(self._basic_variables):
                if i == pivot_i and j == pivot_j:
                    # Pivot element.
                    gamma_new[i, j] = 1 / pivot_gamma
                elif i == pivot_i:
                    # Pivot row.
                    gamma_new[i, j] = self._gamma[pivot_i, j] / pivot_gamma
                elif j == pivot_j:
                    # Pivot column.
                    gamma_new[i, j] = -self._gamma[i, pivot_j] / pivot_gamma
                else:
                    # Other.
                    gamma_new[i, j] = self._gamma[i, j] - self._gamma[pivot_i, j] * self._gamma[i, pivot_j] / pivot_gamma

        rhs_new = self._rhs.copy()
        for i, non_basic_var in enumerate(self._non_basic_variables):
            if i == pivot_i:
                # Pivot row.
                rhs_new[i] = (pivot_rhs / pivot_gamma) % self._time_period
            else:
                # Other.
                rhs_new[i] = (self._rhs[i] - pivot_rhs * self._gamma[i, pivot_j] / pivot_gamma) % self._time_period

        self._basic_variables[pivot_j] = pivot_non_basic_variable
        self._non_basic_variables[pivot_i] = pivot_basic_variable
        self._gamma = gamma_new
        self._rhs = rhs_new

        return pivot_cost_change


    def _calculate_cost_difference(self, i: int, j: int) -> float:
        """
        Calculates the cost difference that is expected from exchange co-tree arc ``i`` with tree arc ``j``.

        :param int i: Index of the co-tree arc.
        :param int j: Index of the tree arc.
        :return: The expected cost difference of infinity, if applying the exchange would result in a division by zero.
        :rtype: float
        """

        ai = self._non_basic_variables[i]
        aj = self._basic_variables[j]
        w_i = float(self._weights.get_named_value(ai))
        w_j = float(self._weights.get_named_value(aj))
        b_i = float(self._rhs[i])
        g_ij = self._gamma[i, j]
        if g_ij == 0:
            return np.Infinity

        delta_cost = w_i * b_i - w_j * ((b_i / g_ij) % self._time_period)
        for k, ak in enumerate(self._non_basic_variables):
            if k != i:
                w_k = float(self._weights.get_named_value(ak))
                b_k = float(self._rhs[k])
                g_kj = self._gamma[k, j]

                delta_cost += w_k * (b_k - ((b_k - b_i * g_kj / g_ij) % self._time_period))

        # The above calculates the negative delta-cost. It implements formula (9) of the paper but the derivation
        # contains errors and results in :math:`\omega - \tilde{\omega}_{ij}` rather than the
        # correct :math:`\omega - \tilde{\omega}_{ij}`.
        return -delta_cost


    def _calculate_cost(self) -> float:
        """
        Calculates the current cost of the tableau.

        :return: The cost.
        :rtype: float
        """

        return float(self._weights.get_sub_vector(self._non_basic_variables).T @ self._rhs)


    def _print_simplex_tableau(self, cost: Optional[float] = None) -> None:
        """
        Prints the simplex tableau to stdout.
        """

        print(np.hstack([self._gamma, self._rhs, self._weights.get_sub_vector(self._non_basic_variables)]))
        if cost is None:
            cost = self._calculate_cost()
        print('Cost: %.2f' % cost)

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import wandb

import wandb_util
from graph_util import construct_edge_cycle_matrix, find_complement_edges
from modulo_simplex_pivoting import ModuloSimplexPivoting
from util import EdgeMatrix, EdgeVector, EventNetworkEdge, IS_EXPERIMENT, NodeVector



class ImprovingCut:
    #: Whether there is an improving cut available.
    available: bool = False
    #: The single node that generated the single node cut.
    node: Optional[str]
    #: The current improving cut vector.
    vector: Optional[EdgeVector]
    #: The delta of the cut.
    delta: float


    def __repr__(self):
        return 'available: %s; node: %s; vector: %s; delta: %s' % (
                repr(self.available), repr(self.node), repr(self.vector), repr(self.delta))


    def __str__(self):
        if self.available:
            return 'Cut available, node: %s; vector: %s; delta: %s' % (str(self.node), str(self.vector), str(self.delta))
        return 'No cut available.'



class ModuloNetworkSimplex:
    #: Whether the algorithm should run verbose (i.e. print out its status).
    _verbose: bool
    #: Stores the current iteration (used for logging to wandb).
    _iteration: int = 0

    #: The global time period (:math:`T`).
    _time_period: int
    #: The nodes of the network.
    _nodes: List[str]
    #: The edges of the network.
    _edges: List[EventNetworkEdge]

    #: The initial (and unchanging) directed event network graph.
    _network: nx.DiGraph

    #: Vector of the lower bounds. If ``_tree_edges`` and ``_complement_edges`` are set and ``_separate_vectors`` was invoked,
    #: this is separated into tree and co-tree edges.
    _lower_bounds: EdgeVector
    #: Vector of the lower bounds, ordered the same as ``_lower_bounds``.
    _upper_bounds: EdgeVector
    #: The difference vector :math:`\vec{\delta} = \vec{u} - \vec{l}`, ordered the same as ``_lower_bounds``.
    _deltas: EdgeVector

    #: The weights of all edges, ordered the same as ``_lower_bounds``.
    _weights: EdgeVector

    #: Event times (:math:`\pi`) for all nodes. This vector will contain the result of the modulo network simplex execution.
    _event_times: NodeVector
    #: The slack times (y) for all edges, ordered the same as _lower_bounds.
    _slack_times: EdgeVector

    #: The tree edges of the current spanning tree structure.
    _tree_edges: List[Tuple[str, str]]
    #: The complement edges (edges that belong to the event network but not to the spanning tree).
    _complement_edges: List[Tuple[str, str]]
    #: Edge cycle matrix (:math:`\Gamma`) of the current spanning tree structure induced by ``_tree_edges``, separated by the tree
    #: and co-tree edges.
    _edge_cycle_matrix: EdgeMatrix

    #: Contains the information whether an improving cut is possible and, if one is possible, which node and edges are part of it.
    _cut: ImprovingCut

    #: Contains the modulo parameters that have been fixed to solve the non-periodic dual.
    _modulo_parameters: EdgeVector


    def __init__(self, network_file: str = 'network-data.mns', verbose: bool = False):
        self._verbose = verbose

        self._read_network_from_file(network_file)

        if IS_EXPERIMENT:
            wandb.config.network_file = network_file
            wandb.config.verbose = verbose

        # Initialize all values derived from the read data.
        self._lower_bounds = EdgeVector(self._edges)
        self._upper_bounds = EdgeVector(self._edges)
        self._deltas = EdgeVector(self._edges)
        self._weights = EdgeVector(self._edges)
        for it_edge in self._edges:
            self._lower_bounds.set_named_value(it_edge, it_edge.lower_bound)
            self._upper_bounds.set_named_value(it_edge, it_edge.upper_bound)
            self._deltas.set_named_value(it_edge, it_edge.upper_bound - it_edge.lower_bound)
            self._weights.set_named_value(it_edge, it_edge.weight)

        # Initialize the values to be optimized.
        self._event_times = NodeVector(self._nodes)
        self._slack_times = EdgeVector(self._edges)

        # Initialize the network graph
        self._network = nx.DiGraph()
        self._network.add_nodes_from(self._nodes)
        self._network.add_edges_from(map(lambda x: x.basic_edge(), self._edges))

        # Other instance variables needed for the optimization.
        self._cut = ImprovingCut()


    def solve(self) -> None:
        print('Starting modulo network simplex algorithm.')

        self._find_initial_feasible_solution()

        # On the start, no cut is available so guarantee that at least one iteration runs.
        cost = -1
        while (self._iteration == 0 or self._cut.available) and self._iteration < 10:
            self._iteration += 1
            wandb_util.uber_iteration += 1

            if self._cut.available:
                self._apply_cut()
                self._solve_non_periodic()

                if IS_EXPERIMENT:
                    wandb.log({ 'cost': cost }, step = wandb_util.uber_iteration)

            cost = self._perform_pivoting()

            self._search_improving_cut()

        print('Terminating.')

        self._calculate_event_times()

        print('\nResult (cost %.2f):' % cost)
        print('Nodes:           ', self._nodes)
        print('Tree edges:      ', self._tree_edges)
        print('Complement edges:', self._complement_edges)
        print('Slack times:     ', self._slack_times.get_vector().T)
        print('Event times:     ', self._event_times)


    def _perform_pivoting(self) -> float:
        gamma = self._edge_cycle_matrix.get_matrix().copy()
        rhs = self._slack_times.get_sub_vector(self._complement_edges).copy()

        cost = ModuloSimplexPivoting(self._time_period, self._tree_edges, self._complement_edges, gamma, rhs, self._weights,
                                     self._verbose).perform_pivoting()
        self._separate_vectors()
        self._edge_cycle_matrix.set_matrix(gamma)

        for it_edge in self._tree_edges:
            self._slack_times.set_named_value(it_edge, 0)
        for it_edge, value in zip(self._complement_edges, rhs):
            self._slack_times.set_named_value(it_edge, value)

        return cost


    def _solve_non_periodic(self) -> None:
        graph = nx.DiGraph()
        for it_node in self._nodes:
            demand = 0
            for it_edge in self._network.in_edges(it_node):
                demand += self._weights.get_named_value(it_edge)
            for it_edge in self._network.out_edges(it_node):
                demand -= self._weights.get_named_value(it_edge)
            graph.add_node(it_node, demand = demand)

        for it_edge in self._edges:
            modulo_parameter = self._modulo_parameters.get_named_value(it_edge)
            lower_bound = self._lower_bounds.get_named_value(it_edge)
            upper_bound = self._upper_bounds.get_named_value(it_edge)
            graph.add_edge(it_edge.source, it_edge.target, weight = (modulo_parameter * self._time_period - lower_bound))
            graph.add_edge(it_edge.target, it_edge.source, weight = (upper_bound - modulo_parameter * self._time_period))

        flow_dict = nx.min_cost_flow(graph)

        # Build the tree.
        spanning_tree = nx.DiGraph()
        spanning_tree.add_nodes_from(graph)
        for it_edge in self._edges:
            if flow_dict[it_edge.source][it_edge.target] != 0 or flow_dict[it_edge.target][it_edge.source] != 0:
                spanning_tree.add_edge(it_edge.source, it_edge.target)
        self._build_simplex_tableau(spanning_tree)


    def _search_improving_cut(self) -> None:
        for it_node in self._network.nodes():
            changes = []
            deltas = []
            for delta in range(0, self._time_period):
                changes.append(self._validate_single_node_cut(it_node, delta))
                deltas.append(delta)

            min_change_index = int(np.argmin(changes))
            min_change = changes[min_change_index]
            delta = deltas[min_change_index]
            if min_change < 0:
                # Build the cut vector.
                cut = EdgeVector(self._network.edges())
                for it_edge in self._network.in_edges(it_node):
                    cut.set_named_value(it_edge, -1)
                for it_edge in self._network.out_edges(it_node):
                    cut.set_named_value(it_edge, 1)

                self._cut.available = True
                self._cut.node = it_node
                self._cut.vector = cut
                self._cut.delta = delta

                if self._verbose:
                    print('Found an improving cut:', self._cut)

                return

        self._cut.available = False
        if self._verbose:
            print('No improving cut found.')


    def _validate_single_node_cut(self, node: str, delta: int) -> float:
        change = 0

        for it_edge in self._network.in_edges(node):
            slack_time = self._slack_times.get_named_value(it_edge)
            lower_bound = self._lower_bounds.get_named_value(it_edge)
            upper_bound = self._upper_bounds.get_named_value(it_edge)

            tension = slack_time + lower_bound
            tension += delta
            if tension > upper_bound:
                tension -= self._time_period
            if tension < lower_bound:
                # Node cut with delta is not feasible.
                return np.Infinity
            change += self._weights.get_named_value(it_edge) * (tension - lower_bound - slack_time)

        for it_edge in self._network.out_edges(node):
            slack_time = self._slack_times.get_named_value(it_edge)
            lower_bound = self._lower_bounds.get_named_value(it_edge)
            upper_bound = self._upper_bounds.get_named_value(it_edge)

            tension = slack_time + lower_bound
            tension -= delta
            if tension < lower_bound:
                tension += self._time_period
            if tension > upper_bound:
                # Node cut with delta is not feasible.
                return np.Infinity
            change += self._weights.get_named_value(it_edge) * (tension - lower_bound - slack_time)

        return change


    def _apply_cut(self) -> None:
        self._calculate_modulo_parameters()

        if self._verbose:
            print('Applying the cut. Slack and modulo before:', self._slack_times, self._modulo_parameters)

        for it_edge in self._network.edges():
            lower_bound = self._lower_bounds.get_named_value(it_edge)
            upper_bound = self._upper_bounds.get_named_value(it_edge)

            tension = self._slack_times.get_named_value(it_edge) + lower_bound

            cut_direction = self._cut.vector.get_named_value(it_edge)
            tension = tension + cut_direction * self._cut.delta
            if cut_direction == -1:
                if tension > upper_bound:
                    tension -= self._time_period
                    self._modulo_parameters.set_named_value(it_edge, self._modulo_parameters.get_named_value(it_edge) - 1)
                self._slack_times.set_named_value(it_edge, tension - lower_bound)
            elif cut_direction == 1:
                if tension < lower_bound:
                    tension += self._time_period
                    self._modulo_parameters.set_named_value(it_edge, self._modulo_parameters.get_named_value(it_edge) + 1)
                self._slack_times.set_named_value(it_edge, tension - lower_bound)

        self._cut.available = False

        if self._verbose:
            print('Cut applied. Slack and modulo after:      ', self._slack_times, self._modulo_parameters)


    def _calculate_modulo_parameters(self) -> None:
        self._calculate_event_times()

        # For all tree edges, the modulo parameter is zero (which is automatically implied as the constructor of an edge vector
        # sets all elements to zero).
        self._modulo_parameters = EdgeVector(self._network.edges())
        for it_edge in self._complement_edges:
            (source, target) = it_edge
            lower_bound = self._lower_bounds.get_named_value(it_edge)
            upper_bound = self._upper_bounds.get_named_value(it_edge)

            tension = self._event_times.get_named_value(target) - self._event_times.get_named_value(source)
            # Manually apply the modulus operator.
            modulo_parameter = 0
            while tension > upper_bound:
                tension -= self._time_period
                modulo_parameter -= 1
            while tension < lower_bound:
                tension += self._time_period
                modulo_parameter += 1

            self._modulo_parameters.set_named_value(it_edge, modulo_parameter)

            if self._verbose:
                print('Edge', it_edge, 'has modulo parameter', modulo_parameter)


    def _calculate_event_times(self) -> None:
        # Delete all the current times by setting them to -1.
        self._event_times.set_vector(-np.ones(self._event_times.get_vector().shape))

        # Pick some start node and set its event time to 0. Then proceed recursively.
        start = self._nodes[0]
        self._event_times.set_named_value(start, 0)
        self._calculate_event_times_rec(start)


    def _calculate_event_times_rec(self, node):
        node_potential = self._event_times.get_named_value(node)
        for it_edge in self._tree_edges:
            (source, target) = it_edge
            lower_bound = self._lower_bounds.get_named_value(it_edge)

            if source == node:
                # Outgoing edge.
                if self._event_times.get_named_value(target) == -1:
                    self._event_times.set_named_value(target, node_potential + lower_bound)
                    self._calculate_event_times_rec(target)

            elif target == node:
                # Incoming edge.
                if self._event_times.get_named_value(source) == -1:
                    self._event_times.set_named_value(source, node_potential - lower_bound)
                    self._calculate_event_times_rec(source)


    def _find_initial_feasible_solution(self) -> None:
        # Warning: This does not keep the arc directions!
        graph = nx.Graph()
        graph.add_nodes_from(self._network)
        graph.add_edges_from(self._network.edges())

        # TODO: Switch to minimal spanning tree!
        spanning_tree = nx.minimum_spanning_tree(graph)
        # spanning_tree = nx.DiGraph()
        # spanning_tree.add_nodes_from(graph)
        # spanning_tree.add_edges_from([('B', 'C'), ('D', 'A'), ('D', 'C')])
        self._build_simplex_tableau(spanning_tree)


    def _build_simplex_tableau(self, spanning_tree: nx.DiGraph) -> None:
        # Contains the correct directions of the edges.
        (self._tree_edges, self._complement_edges) = find_complement_edges(self._network, spanning_tree)
        self._edge_cycle_matrix = construct_edge_cycle_matrix(self._network, self._tree_edges, self._complement_edges)
        self._separated_edges = self._tree_edges + self._complement_edges

        self._separate_vectors()

        # Calculate the basic solution.
        b = (-self._edge_cycle_matrix.get_matrix() @ self._lower_bounds.get_vector()) % self._time_period

        # TODO: Maybe check whether the basic solution is feasible? Just hope for now.

        # Copy the basic solution into the slack_times vector.
        for edge in self._tree_edges:
            self._slack_times.set_named_value(edge, 0)
        for i, edge in enumerate(self._complement_edges):
            self._slack_times.set_named_value(edge, float(b[i]))

        if self._verbose:
            print('Found a basic solution:', self._slack_times.get_vector().T, 'with tree edges', self._tree_edges,
                  'and complement edges', self._complement_edges)


    def _separate_vectors(self) -> None:
        separated_edge_cycle_matrix = EdgeMatrix(self._complement_edges, self._separated_edges)
        separated_lower_bounds = EdgeVector(self._separated_edges)
        separated_upper_bounds = EdgeVector(self._separated_edges)
        separated_deltas = EdgeVector(self._separated_edges)
        separated_weights = EdgeVector(self._separated_edges)
        separated_slack_times = EdgeVector(self._separated_edges)
        for it_edge in self._separated_edges:
            separated_edge_cycle_matrix.set_named_col(it_edge, self._edge_cycle_matrix.get_named_col(it_edge))
            separated_lower_bounds.set_named_value(it_edge, self._lower_bounds.get_named_value(it_edge))
            separated_upper_bounds.set_named_value(it_edge, self._upper_bounds.get_named_value(it_edge))
            separated_deltas.set_named_value(it_edge, self._deltas.get_named_value(it_edge))
            separated_weights.set_named_value(it_edge, self._weights.get_named_value(it_edge))
            separated_slack_times.set_named_value(it_edge, self._slack_times.get_named_value(it_edge))
        self._edge_cycle_matrix = separated_edge_cycle_matrix
        self._lower_bounds = separated_lower_bounds
        self._upper_bounds = separated_upper_bounds
        self._deltas = separated_deltas
        self._weights = separated_weights
        self._slack_times = separated_slack_times


    def _read_network_from_file(self, file: str) -> None:
        """
        Reads the network data from the file with the given name.

        :param str file: The path to the network data file.
        """

        self._time_period = 0
        self._nodes = []
        self._edges = []
        with open(file, 'r') as fd:
            reading_time_period = False
            reading_nodes = False
            reading_edges = False
            for line in fd:
                line = line.strip()

                # Ignore empty lines and comments.
                if line == '' or line.startswith('#'):
                    continue

                if line == 'T':
                    reading_time_period = True
                    reading_nodes = False
                    reading_edges = False
                elif line == 'nodes':
                    reading_time_period = False
                    reading_nodes = True
                    reading_edges = False
                elif line == 'edges':
                    reading_time_period = False
                    reading_nodes = False
                    reading_edges = True

                elif reading_time_period:
                    self._time_period = int(line.strip())
                elif reading_nodes:
                    self._nodes.append(line)
                elif reading_edges:
                    edge_data = list(map(lambda x: x.strip(), line.split(',')))
                    self._edges.append(EventNetworkEdge(
                            source = edge_data[0],
                            target = edge_data[1],
                            weight = float(edge_data[2]),
                            lower_bound = int(edge_data[3]),
                            upper_bound = int(edge_data[4])))



if __name__ == '__main__':
    ModuloNetworkSimplex('network-data.mns', verbose = True).solve()

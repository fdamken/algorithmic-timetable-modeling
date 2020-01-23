from typing import List, Tuple, Optional

import networkx as nx
import numpy as np

from graph_util import construct_edge_cycle_matrix, find_complement_edges
from modulo_simplex_pivoting import ModuloSimplexPivoting
from util import EdgeMatrix, EdgeVector, EventNetworkEdge, NodeVector



class ImprovingCut:
    #: Whether there is an improving cut available.
    available: bool
    #: The single node that generated the single node cut.
    cut_node: Optional[str]
    #: The current improving cut vector.
    cut_vector: Optional[EdgeVector]


    def __init__(self):
        self.available = False
        self.cut_node = ''
        self.cut_vector = None



class ModuloNetworkSimplex:
    #: Whether the algorithm should run verbose (i.e. print out its status).
    _verbose: bool
    #: Stores the current iteration (used for logging to wandb).
    _iteration: int

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


    def __init__(self, network_file: str = 'network-data.mns', verbose: bool = False):
        self._verbose = verbose

        self._read_network_from_file(network_file)

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
        # TODO: See modulo_network_simplex.cpp, void simplex::solve().

        self._find_initial_feasible_solution()

        # On the start, no cut is available so guarantee that at least one iteration runs.
        while self._iteration == 0 or self._cut.available:
            if self._cut.available:
                # TODO: Transform, solve non-periodic and build tableau.
                pass

            self._perform_pivoting()

            self._search_improving_cut()


    def _perform_pivoting(self) -> None:
        gamma = self._edge_cycle_matrix.get_matrix().copy()
        rhs = self._slack_times.get_sub_vector(self._complement_edges).copy()
        ModuloSimplexPivoting(self._time_period, self._tree_edges, self._complement_edges, gamma, rhs, self._weights,
                              self._verbose).perform_pivoting()
        self._separate_vectors()
        self._edge_cycle_matrix.set_matrix(gamma)
        for it_edge in self._tree_edges:
            self._slack_times.set_named_value(it_edge, 0)
        for it_edge, value in zip(self._complement_edges, rhs):
            self._slack_times.set_named_value(it_edge, value)


    def _solve_non_periodic(self) -> None:
        # TODO: See modulo_network_simplex.cpp, void simplex::non_periodic().
        pass


    def _search_improving_cut(self) -> None:
        # TODO: See modulo_network_simplex.cpp, void simplex::improvable() (restricted to local_search == SINGLE_NODE_CUT).
        pass


    def _apply_cut(self) -> None:
        # TODO: See modulo_network_simplex.cpp, void simplex::transform().
        pass


    def _calculate_modulo_parameters(self) -> None:
        # TODO: See modulo_network_simplex.cpp, void simplex::find_modulo().
        pass


    def _calculate_event_times(self) -> None:
        # TODO: See modulo_network_simplex.cpp, void simplex::set_time() and void simplex::set_time(Vertex where).
        pass


    def _find_initial_feasible_solution(self) -> None:
        # Warning: This does not keep the arc directions!
        graph = nx.Graph()
        graph.add_nodes_from(self._network)
        graph.add_edges_from(self._network.edges())

        # TODO: Switch to minimal spanning tree!
        # spanning_tree = nx.minimum_spanning_tree(graph)
        spanning_tree = nx.DiGraph()
        spanning_tree.add_nodes_from(graph)
        spanning_tree.add_edges_from([('B', 'C'), ('D', 'A'), ('D', 'C')])
        # Contains the correct directions of the edges.
        (self._tree_edges, self._complement_edges) = find_complement_edges(self._network, spanning_tree)
        self._edge_cycle_matrix = construct_edge_cycle_matrix(self._network, self._tree_edges, self._complement_edges)
        self._separated_edges = self._tree_edges + self._complement_edges

        self._separate_vectors()

        # Calculate the periodic basic solution.
        b = (-self._edge_cycle_matrix.get_matrix() @ self._lower_bounds.get_vector()) % self._time_period

        # TODO: Maybe check whether the periodic basic solution is feasible? Just hope for now.

        # Copy the periodic basic solution into the slack_times vector.
        for edge in self._tree_edges:
            self._slack_times.set_named_value(edge, 0)
        for i, edge in enumerate(self._complement_edges):
            self._slack_times.set_named_value(edge, float(b[i]))

        if self._verbose:
            print('Found a periodic basic solution:', self._slack_times.get_vector().T, 'with tree edges', self._tree_edges,
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

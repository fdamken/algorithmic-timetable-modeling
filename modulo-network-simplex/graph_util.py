import networkx as nx
import numpy as np
from util import EdgeMatrix


def find_complement_edges(network: nx.DiGraph, tree: nx.DiGraph):
    """
    Finds the complement edges (the edges that are present in the network, but not in the tree) of the network/tree combination.

    :param nx.DiGraph network: The network.
    :param nx.DiGraph tree: The tree.
    :return: A tuple ``(tree_edges, complement_edges)`` with the tree and the co-tree edges (both with the original orientation).
    :rtype: (list, list)
    """

    tree_edges = []
    complement_edges = []
    for edge in network.edges():
        if edge in tree.edges() or tuple(reversed(edge)) in tree.edges():
            tree_edges.append(edge)
        else:
            complement_edges.append(edge)
    return tree_edges, complement_edges


def construct_edge_cycle_matrix(network: nx.DiGraph, tree_edges: list, complement_edges: list):
    """
    Builds the edge cycle matrix for the given network and the given tree/co-tree edges.

    :param nx.DiGraph network: The network with the original structure.
    :param list tree_edges: All tree edges (e.g. as returned by ``find_complement_edges``).
    :param list complement_edges: All co-tree edges (e.g. as returned by ``find_complement_edges``).
    :return: The edge cycle matrix with the co-tree arcs as ordered in ``complement_edges`` and the edges as ordered in ``network``.
    :rtype: EdgeMatrix
    """

    result = EdgeMatrix(complement_edges, network.edges())
    i = 0
    for complement_edge in complement_edges:
        graph = nx.Graph()
        graph.add_nodes_from(network)
        graph.add_edge(*complement_edge)
        graph.add_edges_from(tree_edges)
        # This must be exacle one as adding a co-tree arc to the tree defines a unique cycle.
        cycle = nx.cycle_basis(graph)[0]

        incidence_vector = np.zeros(result.get_matrix().shape[1])
        edge_cycle_semi_positive_value = 1
        previous = cycle[-1]
        for node in cycle:
            cycle_edge = (previous, node)

            if cycle_edge == complement_edge:
                edge_cycle_semi_positive_value = 1
            elif tuple(reversed(cycle_edge)) == complement_edge:
                # We have to invert everything. Only at this point we know that
                # we have interpreted the cycle in the wrong direction, so
                # every "positive" direction really was a negative.
                incidence_vector = -incidence_vector
                edge_cycle_semi_positive_value = -1
            previous = node

            for network_edge in network.edges():
                if cycle_edge == network_edge:
                    # Same direction as the complement edge.
                    incidence_vector[result.get_col_index(network_edge)] = edge_cycle_semi_positive_value
                elif tuple(reversed(cycle_edge)) == network_edge:
                    # Other direction as the edge_cycle_semi_positive_value edge.
                    incidence_vector[result.get_col_index(network_edge)] = -edge_cycle_semi_positive_value

        result.set_named_row(complement_edge, incidence_vector)

        i += 1
    return result

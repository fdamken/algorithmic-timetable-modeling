import networkx as nx
import numpy as np

from graph_util import construct_edge_cycle_matrix, find_complement_edges
from modulo_simplex_pivoting import ModuloSimplexPivoting
from util import EdgeMatrix, EdgeVector, NodeVector


# Load the network data from the files variables.csv, graph-nodes.csv and graph-edges.csv.
variables = np.genfromtxt('variables.csv', delimiter = ',', skip_header = 1)
time_period = int(variables)
nodes = np.genfromtxt('graph-nodes.csv', dtype = str, delimiter = ',', skip_header = 1)
edges = list(map(lambda x: (x[0], x[1], {
        'weight': float(x[2]),
        'lower_bound': int(x[3]),
        'upper_bound': int(x[4])
}), np.genfromtxt('graph-edges.csv', dtype = str, delimiter = ',', skip_header = 1)))

# Initialize all constant values from the network data.
lower_bounds = EdgeVector(edges)
upper_bounds = EdgeVector(edges)
deltas = EdgeVector(edges)
for edge in edges:
    attributes = edge[2]
    lower_bound = attributes['lower_bound']
    upper_bound = attributes['upper_bound']
    lower_bounds.set_named_value(edge, lower_bound)
    upper_bounds.set_named_value(edge, upper_bound)
    deltas.set_named_value(edge, upper_bound - lower_bound)

# Initialize the values to optimize.
event_times = NodeVector(nodes)
slack_times = EdgeVector(edges)

#
##################
# INITIALIZATION #
##################

network = nx.DiGraph()
network.add_nodes_from(nodes)
network.add_edges_from(edges)

# Warning: This does not keep the arc directions!
graph = nx.Graph()
graph.add_nodes_from(network)
graph.add_edges_from(network.edges())

# spanning_tree = nx.minimum_spanning_tree(graph)
spanning_tree = nx.DiGraph()
spanning_tree.add_nodes_from(graph)
spanning_tree.add_edges_from([('B', 'C'), ('D', 'A'), ('D', 'C')])
# Contains the correct directions of the edges.
(tree_edges, complement_edges) = find_complement_edges(network, spanning_tree)
edge_cycle_matrix = construct_edge_cycle_matrix(network, tree_edges, complement_edges)

separated_edges = tree_edges + complement_edges

separated_edge_cycle_matrix = EdgeMatrix(complement_edges, separated_edges)
separated_lower_bounds = EdgeVector(separated_edges)
for edge in separated_edges:
    separated_edge_cycle_matrix.set_named_col(edge, edge_cycle_matrix.get_named_col(edge))
    separated_lower_bounds.set_named_value(edge, lower_bounds.get_named_value(edge))

# Calculate the periodic basic solution.
b = (-separated_edge_cycle_matrix.get_matrix() @ separated_lower_bounds.get_vector()) % time_period

# TODO: Maybe check iff the periodic basic solution is feasible? Just hope for now.

# Copy the periodic basic solution into the slack_times vector.
for edge in tree_edges:
    slack_times.set_named_value(edge, 0)
for i, edge in enumerate(complement_edges):
    slack_times.set_named_value(edge, float(b[i]))

edge_weights = nx.get_edge_attributes(network, 'weight')
weights = EdgeVector(edges)
for edge in edges:
    weights.set_named_value(edge, edge_weights[(edge[0], edge[1])])
cost = float(weights.get_sub_vector(complement_edges).T @ b)

#
#############
# ALGORITHM #
#############

basic_variables = tree_edges.copy()
non_basic_variables = complement_edges.copy()
gamma = separated_edge_cycle_matrix.get_matrix().copy()
rhs = b.copy()

ModuloSimplexPivoting(time_period, basic_variables, non_basic_variables, gamma, rhs, weights).perform_pivoting(log = True)
# Copy the right-hand-side back to the slack times.
for edge in basic_variables:
    slack_times.set_named_value(edge, 0)
for (edge, value) in zip(non_basic_variables, rhs):
    slack_times.set_named_value(edge, value)



# TODO: How on earth should the single node cutting be done???
def apply_improving_single_node_cut():
    for node in network.nodes():
        for delta in range(0, time_period):
            snc_change = 0
            improved_slack_times = slack_times.get_vector().copy()
            for edge in network.in_edges(node):
                snc_change += weights.get_named_value(edge) * (slack_times.get_named_value(edge) - delta)
                improved_slack_times[slack_times.get_index(edge)] += delta
            for edge in network.out_edges(node):
                snc_change += weights.get_named_value(edge) * (slack_times.get_named_value(edge) + delta)
                improved_slack_times[slack_times.get_index(edge)] -= delta

            # TODO: This does not really make sense...
            if snc_change < 0 and (0 <= improved_slack_times).all() and (improved_slack_times <= deltas.get_vector()).all():
                slack_times.set_vector(improved_slack_times)
                return True
    return False



print('slack_times', slack_times.get_vector().T)
success = apply_improving_single_node_cut()
print('SNC success:', success)
print('slack_times', slack_times.get_vector().T)

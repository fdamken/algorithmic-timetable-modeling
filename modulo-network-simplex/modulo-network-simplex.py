import networkx as nx
import numpy as np

from util import NodeVector, EdgeVector, EdgeMatrix


def find_complement_edges(network, tree):
    tree_edges = []
    complement_edges = []
    for edge in network.edges():
        if edge in tree.edges() or tuple(reversed(edge)) in tree.edges():
            tree_edges.append(edge)
        else:
            complement_edges.append(edge)
    return (tree_edges, complement_edges)


def construct_edge_cycle_matrix(network, tree_edges, complement_edges):
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


# Load the network data from the files variables.csv, graph-nodes.csv and graph-edges.csv.
variables = np.genfromtxt('variables.csv', delimiter=',', skip_header=1)
T = variables
nodes = np.genfromtxt('graph-nodes.csv', dtype=str, delimiter=',', skip_header=1)
edges = list(map(lambda x: (x[0], x[1], {
    'weight': float(x[2]),
    'lower_bound': int(x[3]),
    'upper_bound': int(x[4])
}),
                 np.genfromtxt('graph-edges.csv', dtype=str, delimiter=',', skip_header=1)))

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
b = (-separated_edge_cycle_matrix.get_matrix() @ separated_lower_bounds.get_vector()) % T

# TODO: Maybe check iff the periodic basic solution is feasible? Just hope for now.

# Copy the periodic basic solution into the slack_times vector.
for edge in tree_edges:
    slack_times.set_named_value(edge, 0)
for i, edge in enumerate(complement_edges):
    slack_times.set_named_value(edge, float(b[i]))

edge_weights = nx.get_edge_attributes(network, 'weight')
weights = np.array(list(map(lambda edge: edge_weights[edge], complement_edges))).reshape(-1, 1)
cost = float(weights.T @ b)

print('Found a periodic basic solution y =', slack_times.get_vector().reshape(1, -1), 'with cost =', cost)

#############
# ALGORITHM #
#############

basic_variables = tree_edges.copy()
non_basic_variables = complement_edges.copy()
gamma = separated_edge_cycle_matrix.get_matrix().copy()
rhs = b.copy()
weights = weights.copy()

print(gamma)
print(rhs)
print(weights)


def print_tableau():
    print(np.hstack([gamma, rhs, weights]))


def calculate_cost_difference(i, j):
    w_i = float(weights[i])
    w_j = float(weights[j])
    b_i = float(rhs[i])
    g_ij = gamma[i, j]
    if g_ij == 0:
        return np.Infinity

    delta_cost = w_i * b_i - w_j * ((b_i / g_ij) % T)
    for k in range(0, len(basic_variables)):
        if k != i:
            w_k = float(weights[k,])
            b_k = float(rhs[k])
            g_kj = gamma[k, j]

            print('i:', i, '  j:', j, '  k:', k, '  w_k:', w_k, '  b_k:', b_k, '  g_kj:', g_kj, '  g_ij:', g_ij)
            delta_cost += w_k * (b_k - ((b_k - b_i * g_kj / g_ij) % T))
    return delta_cost


def perform_modulo_simplex_pivoting_step():
    cost_change_matrix = np.zeros((len(non_basic_variables), len(basic_variables)))
    for i, non_basic_edge in enumerate(non_basic_variables):
        for j, basic_edge in enumerate(basic_variables):
            print()
            cost_change_matrix[i, j] = calculate_cost_difference(i, j)
            print('Exchange', non_basic_edge, 'with', basic_edge, '--> weight change:', cost_change_matrix[i, j])
    print(non_basic_variables)
    print(basic_variables)
    print(cost_change_matrix)


def perform_modulo_simplex_pivoting():
    pass


print(gamma)
perform_modulo_simplex_pivoting_step()

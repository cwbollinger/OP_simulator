#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 15:25:33 2018
@author: sritee
"""

# Orienteering problem with Miller-Tucker-Zemlin formulation

import cvxpy as c
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt


def build_graph(path_solution, node_scores, edge_costs):
    g = nx.DiGraph()

    verified_cost = 0
    now_node = 0  # Initialize at start node
    tour = []

    # g.add_nodes_from(range(1,num_nodes+1))
    for k in range(edge_costs.shape[0]):
        g.add_node(k, value=node_scores[k])  # 1 based indexing

    while(True):  # till we reach end node
        tour.append(now_node)

        next_node = np.argmax(path_solution.value[now_node, :])  # where we go from node i
        # 1 based indexing graph
        g.add_edge(now_node, next_node, weight=int(edge_costs[now_node, next_node]))
        # build up the cost
        verified_cost += edge_costs[now_node, next_node]
        # for 1 based indexing
        now_node = next_node
        # we have looped again
        if next_node == 0:
            break

    return g, tour, verified_cost


def display_graph(g, save_name='foo'):
    color_map = ['red'] * num_nodes
    color_map[0] = 'green'

    pos = nx.circular_layout(g)
    nodeval = nx.get_node_attributes(g, 'value')
    nx.draw_circular(g, with_labels=True, node_color=color_map, node_size=1000, labels=nodeval)
    labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, width=20, edge_color='b')
    plt.savefig('{}.png'.format(save_name))
    plt.title('Time horizon {}'.format(total_time))
    plt.show()


def get_solution(score_vector, cost_matrix):
    plan_idxs = c.Variable(cost_matrix.shape, boolean=True)

    # variables in subtour elimination constraints
    u = c.Variable(score_vector.shape[0])

    cost = c.trace(cost_matrix.T @ plan_idxs)  # total cost of the tour
    profit = c.sum(plan_idxs @ score_vector)

    ones_arr = np.ones(score_vector.shape)  # array for ones

    constraints = []

    # now, let us make sure each node is visited only once, and we leave only once from that node.

    # we leave from the first node
    constraints.append(c.sum(plan_idxs[0, :]) == 1)
    # we come back to the first node
    constraints.append(c.sum(plan_idxs[:, 0]) == 1)

    # max one connection outgoing and incoming
    constraints.append(plan_idxs.T @ ones_arr <= 1)
    constraints.append(plan_idxs @ ones_arr <= 1)

    for i in range(num_nodes):
        constraints.append(c.sum(plan_idxs[:, i]) == c.sum(plan_idxs[i, :]))

    # let us add the time constraints
    constraints.append(cost <= total_time)

    # Let us add the subtour elimination constraints (Miller-Tucker-Zemlin similar formulation)
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                constraint = u[i] - u[j] + num_nodes * (plan_idxs[i, j] - 1) + 1 <= 0
                constraints.append(constraint)
            else:
                continue

    prob = c.Problem(c.Maximize(profit), constraints)

    prob.solve()

    if np.any(plan_idxs.value is None):  # no feasible solution found!

        print('Feasible solution not found, lower your time constraint!')
        raise ValueError()

    return plan_idxs, prob.value, cost.value, prob.solver_stats.solve_time


np.random.seed(1)

# number of nodes in the Orienteering
num_nodes = 10
# the time horizons which we try out
try_times = range(20, 100, 20)
try_times = [200]

for total_time in try_times:
    # this will generate a random cost matrix.
    cost_matrix = np.random.randint(1, 15, (num_nodes, num_nodes))
    # ensure symmetry of the matrix
    cost_matrix = cost_matrix + cost_matrix.T
    # make sure we don't travel from node to same node, by having high cost.
    np.fill_diagonal(cost_matrix, 1000)

    # this will generate a random score matrix.
    score_vector = np.random.randint(1, 5, (num_nodes))
    # since the 0th node, start node, has no value!
    score_vector[0] = 0

    plan, profit, cost, solve_time = get_solution(score_vector, cost_matrix)

    print('----- Plan -----')
    print(np.around(plan.value, 2).astype('int32'))
    print('----- Edge Costs -----')
    print(cost_matrix)

    g, tour, verified_cost = build_graph(plan, score_vector, cost_matrix)

    msg = 'The maximum profit tour found is \n'
    for idx, k in enumerate(tour):
        msg += str(k)
        if idx < len(tour) - 1:
            msg += ' -> '
        else:
            msg += ' -> 0'
    print(msg)

    print('Profit: {:.2f}, cost: {:.2f}, verification cost: {:.2f} '.format(profit, cost, verified_cost))
    print('Time taken: {:.2f} seconds'.format(solve_time))

    display_graph(g)

# print(cost_matrix)
# print(x.value)
# print(score_vector)

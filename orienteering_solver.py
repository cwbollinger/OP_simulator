#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 15:25:33 2018
@author: sritee
"""

# Orienteering problem with Miller-Tucker-Zemlin formulation
import os

import cvxpy as c
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import load_cost_matrix, build_graph, setup_task_windows, get_constraints, print_constraints_solution

from timeit import default_timer as timer


def display_results(g, tour, costs):
    color_map = ['red'] * num_nodes
    color_map[0] = 'green'

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    pos = nx.circular_layout(g)
    nodeval = nx.get_node_attributes(g, 'value')
    for k, v in nodeval.items():
        nodeval[k] = (k + 1, np.around(nodeval[k], 2))
    nx.draw_circular(g, with_labels=True, node_color=color_map, node_size=1000,
                     labels=nodeval, ax=ax1)
    labels = nx.get_edge_attributes(g, 'weight')
    ax1.set_title('Graph View')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, width=20, edge_color='b', ax=ax1)
    nodes = []
    windows = []
    offsets = []
    # fake_val = task_windows[0][1]  # the start node always has an "open" window
    endval = 0
    for window in task_windows:
        if window[1] > endval:
            endval = window[1]

    for i, task in enumerate(task_windows):
        nodes.append(i + 1)
        if np.isnan(task[1]):
            offsets.append(0)
            windows.append(endval)
        else:
            offsets.append(task[0])
            windows.append(task[1] - task[0])

    ax2.set_title('Time Schedule')
    ax2.barh(nodes, windows, left=offsets, color='b')

    visit_order = tour
    visit_times = [0]
    tour_idx = 0
    time = 0
    for idx in range(len(tour)):
        if idx == len(tour) - 1:
            time += costs[tour[idx], 0]
        else:
            time += costs[tour[idx], tour[idx + 1]]
        visit_times.append(time)
        tour_idx += 1

    visit_order.append(0)
    # print(visit_order)
    # print(visit_times)
    ax2.plot(visit_times, visit_order, 'ro-')
    plt.show()


def get_solution(score_vector, cost_matrix, time_budget=100):

    x = c.Variable(cost_matrix.shape, boolean=True)

    # variables in subtour elimination constraints
    u = c.Variable(score_vector.shape[0], integer=True)

    reward = c.sum(x @ score_vector)

    constraints = get_constraints(cost_matrix, score_vector, x, u)

    # time constraints
    cost = c.sum(c.multiply(cost_matrix, x))  # total cost of the tour
    constraints.append(cost <= budget)

    prob = c.Problem(c.Maximize(reward), constraints)

    print('----- Problem Info -----')
    print('DCP: {}'.format(prob.is_dcp()))
    print('QP: {}'.format(prob.is_qp()))
    print('Mixed Integer: {}'.format(prob.is_mixed_integer()))

    start = timer()
    if prob.is_mixed_integer():
        prob.solve(solver=c.GLPK_MI)
    else:
        prob.solve()
    end = timer()

    if np.any(x.value is None):  # no feasible solution found!
        msg = 'Solver {} failed with Status: {}'.format(
              prob.solver_stats.solver_name, prob.status)
        raise ValueError(msg)

    # print('----- U -----')
    # print(u.value)
    print(vars(prob.solver_stats))
    return x.value, prob.value, cost.value, end - start


def evaluate_solution(costs, rewards, budget):
    x = np.zeros(costs.shape)
    x[0, 3] = 1
    x[3, 0] = 1
    u = np.zeros(costs.shape[0])
    u[0] = 0
    u[3] = 1
    print_constraints_solution(costs, rewards, x, u, budget)


if __name__ == '__main__':
    np.random.seed(1)
    print(c.installed_solvers())
    files = [os.path.join('.', 'Maps', f) for f in os.listdir('Maps')
             if f[-3:] == 'mat' and f[-12:-4] != 'solution']

    for f in files:
        cost_matrix = load_cost_matrix(f)
        # high diagonal costs cause numerical errors
        # use constraints to prevent travel to self
        # diagonal cost must be > 0 for this to work
        # but should be low
        np.fill_diagonal(cost_matrix, 1)
        # cost_matrix = cost_matrix[0:8, 0:8]
        num_nodes = cost_matrix.shape[0]

        score_vector = np.random.randint(1, 5, (num_nodes))
        # since the 0th node, start node, has no value!
        score_vector[0] = 0

        task_windows = setup_task_windows(score_vector)

        budget = 80

        # evaluate_solution(cost_matrix, score_vector, budget)

        plan, reward, cost, solve_time = get_solution(score_vector, cost_matrix, budget)

        print('----- Plan -----')
        print(plan)
        print('----- Edge Costs -----')
        print(cost_matrix)
        print('----- Scores -----')
        print(score_vector)

        g, tour, verified_cost = build_graph(plan, score_vector, cost_matrix)

        msg = 'The maximum reward tour found is \n'
        for idx, k in enumerate(tour):
            msg += str(k)
            if idx < len(tour) - 1:
                msg += ' -> '
            else:
                msg += ' -> 0'
        print(msg)

        print('Profit: {:.2f}, cost: {:.2f}, verification cost: {:.2f}'.format(reward, cost, verified_cost))
        print('Time taken: {:.2f} seconds'.format(solve_time))

        display_results(g, tour, cost_matrix)

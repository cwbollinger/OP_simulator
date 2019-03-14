#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cwb
"""

# Orienteering problem with Miller-Tucker-Zemlin formulation
# Service window formulation added by cwb

import os

import cvxpy as c
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import load_cost_matrix, save_solution, build_graph, get_arrive_depart_pairs, setup_task_windows, get_starting_cost


def compute_wait_times(plan, edge_costs, visit_times):
    wait_times = np.zeros(visit_times.shape)
    curr_state = 0
    for i in range(len(visit_times)):
        next_state = np.argmax(plan[curr_state, :])
        cost = edge_costs[curr_state, next_state]
        if next_state == 0:
            wait_times[curr_state] = 0
        else:
            wait_times[curr_state] = visit_times[next_state] - cost - visit_times[curr_state]
        curr_state = next_state
    wait_times[wait_times < 0] = 0
    return wait_times


def display_results(g, plan, task_windows, visit_times, wait_times, total_cost):
    num_nodes = len(visit_times)
    color_map = ['red'] * num_nodes
    color_map[0] = 'green'

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    pos = nx.circular_layout(g)
    nodeval = nx.get_node_attributes(g, 'value')
    for k, v in nodeval.items():
        nodeval[k] = (k, np.around(nodeval[k], 2))
    nx.draw_circular(g, with_labels=True, node_color=color_map, node_size=1000,
                     labels=nodeval, ax=ax1)
    labels = nx.get_edge_attributes(g, 'weight')
    ax1.set_title('Graph View')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, width=20,
                                 edge_color='b', ax=ax1)
    nodes = []
    windows = []
    offsets = []
    # fake_val = task_windows[0][1]  # the start node always has an "open" window
    endval = total_cost
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

    visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, total_cost)

    visit_order_waits = [x+1 for x in visit_order_waits]
    ax2.plot(visit_times_waits, visit_order_waits, 'ro-')
    plt.show()


def get_solution(node_rewards, time_windows, cost_matrix, max_cost=0):
    num_nodes = node_rewards.shape[0]
    x = c.Variable(cost_matrix.shape, boolean=True)

    # variables in subtour elimination constraints
    s = c.Variable(node_rewards.shape, nonneg=True)

    M = 5000.0

    # cost = c.trace(cost_matrix @ x) + c.sum(s)  # total cost of the tour
    mask_vec = np.zeros(num_nodes)
    mask_vec[0] = 1.0
    # y = c.matmul(np.ones((1, num_nodes)), x)
    # cost = c.max(s) + mask_vec @ cost_matrix @ x[:, 0]
    cost = c.max(s) + mask_vec @ cost_matrix @ x[:, 0]
    # cost = c.sum(c.diag(x @ cost_matrix))
    profit = c.sum(x @ node_rewards)

    constraints = []

    # we leave from the first node
    constraints.append(c.sum(x[0, :]) == 1)
    # we come back to the first node
    constraints.append(c.sum(x[:, 0]) == 1)

    ones_arr = np.ones((node_rewards.shape[0], 1))  # array for ones
    # max one connection outgoing and incoming
    constraints.append(x.T @ ones_arr <= 1)
    constraints.append(x @ ones_arr <= 1)

    for i in range(num_nodes):
        constraints.append(c.sum(x[:, i]) == c.sum(x[i, :]))

    constraints.append(cost <= max_cost)

    # constraints.append(s[0] == 0)
    for i in range(num_nodes):
        # constraints.append(time_windows[i, 0] - s[i] + M * y[0, i] <= M)
        constraints.append(time_windows[i, 0] - s[i] + M * c.sum(x[:, i]) <= M)
        if not np.isnan(time_windows[i, 1]):
            constraints.append(s[i] - time_windows[i, 1] + M * c.sum(x[:, i]) <= M)
        for j in range(num_nodes):
            if i == j:
                constraints.append(x[i, j] == 0)
            elif j == 0:
                pass
            else:
                constraints.append(s[i] + cost_matrix[i, j] - s[j] + M * x[i, j] <= M)

    # print('----------------------')
    # for i, constraint in enumerate(constraints):
    #     print(i + 1, constraint, end='\n\n')
    # print('----------------------')

    prob = c.Problem(c.Maximize(profit), constraints)

    prob.solve()

    if np.any(x.value is None):  # no feasible solution found!

        print('Feasible solution not found, lower your time constraint!')
        print(x.value)
        raise ValueError()

    # for i in range(num_nodes):
    #     y[i] = np.sum(x[:, i])

    print('----- Visited -----')
    for i in range(num_nodes):
        print(np.sum(x.value[:, i]))
    print('----- S -----')
    print(np.around(s.value, 2))

    return x.value, s.value, prob.value, cost.value, prob.solver_stats


if __name__ == '__main__':
    np.random.seed(2)

    files = [os.path.join('.', 'Maps', f) for f in os.listdir('Maps')
             if f[-3:] == 'mat' and f[-12:-4] != 'solution']
    print('Files Found: {}'.format(files))
    # problem_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    # small_map_solve_times = {p: [] for p in problem_sizes}
    # large_map_solve_times = {p: [] for p in problem_sizes}
    # for n in problem_sizes:
    success_count = {}
    for window_ratio in [1, 2, 3, 4, 5]:
        success_count[window_ratio] = 10
        for f in files:
            cost_matrix = load_cost_matrix(f)
            # cost_matrix = cost_matrix[0:n, 0:n]
            # print('----- Edge Costs -----')
            # print(cost_matrix)
            num_nodes = cost_matrix.shape[0]

            # this will generate a random score matrix.
            score_vector = np.ones(num_nodes)
            # score_vector[0] = 0
            task_windows = setup_task_windows(score_vector, window_ratio)
            # print('----- Task Windows -----')
            # print(np.around(task_windows, 2).astype('float'))

            max_cost = get_starting_cost(cost_matrix, task_windows)
            try:
                plan, visit_times, profit, cost, solver_stats = get_solution(score_vector, task_windows, cost_matrix, max_cost)
            except ValueError:
                print('Failed with cost {}'.format(max_cost))
                success_count[window_ratio] -= 1
                continue

            wait_times = compute_wait_times(plan, cost_matrix, visit_times)

            visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, cost)
            save_solution(f, visit_order_waits, visit_times_waits, solver_type='tw')

            print('----- Plan -----')
            print(np.around(plan, 2).astype('int32'))
            # print('----- Edge Costs -----')
            # print(cost_matrix)
            print('----- Wait Times -----')
            print(np.around(wait_times, 2))

            g, tour, verified_cost = build_graph(plan, score_vector, cost_matrix)

            print(f)
            msg = 'The maximum profit tour found is \n'
            for idx, k in enumerate(tour):
                msg += str(k)
                if idx < len(tour) - 1:
                    msg += ' -> '
                else:
                    msg += ' -> 0'
            print(msg)

            msg = 'Profit: {:.2f}, cost: {:.2f}, verification cost: {:.2f}'
            print(msg.format(profit, cost, verified_cost))
            print(solver_stats.solve_time)
            print(solver_stats.setup_time)
            msg = 'Time taken: {} seconds'
            time = solver_stats.solve_time + solver_stats.setup_time
            print(msg.format(time))
            # display_results(g, plan, task_windows, visit_times, wait_times, cost)
            # if f[-12:-7] == '20x20':
            #     small_map_solve_times[n].append(time)
            # elif f[-12:-7] == '50x50':
            #     large_map_solve_times[n].append(time)
        success_count[window_ratio] /= 10.0

    print('Success probabilities across constraint ratios')
    print(success_count)

    # print(small_map_solve_times)
    # print(large_map_solve_times)

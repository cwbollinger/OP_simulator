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


def compute_wait_times(plan, edge_costs, tasks, visit_times, final_cost):
    wait_times = np.zeros(visit_times.shape)
    curr_state = 0
    for i in range(len(visit_times)):
        next_state = np.argmax(plan[curr_state, :])
        cost = edge_costs[curr_state, next_state]
        if next_state == 0:
            # departure_time = visit_times[curr_state] + tasks[curr_state, 2]
            # wait_times[curr_state] = final_cost - cost - departure_time
            wait_times[curr_state] = tasks[curr_state, 2]
        else:
            wait_times[curr_state] = visit_times[next_state] - cost - visit_times[curr_state]
        curr_state = next_state

    wait_times[wait_times < 0] = 0
    return wait_times


def display_results(g, plan, task_windows, visit_times, wait_times, total_cost):
    num_nodes = len(visit_times)
    color_map = ['red'] * num_nodes
    color_map[0] = 'green'

    visited = plan @ np.ones((num_nodes, 1))

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
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, width=20, edge_color='b', ax=ax1)
    nodes = []
    visited_nodes = []
    window_durations = []
    window_starts = []
    window_service_starts = []
    window_service_durations = []
    visited = np.reshape(visited, -1)

    endval = total_cost
    for window in task_windows:
        if window[1] > endval:
            endval = window[1]

    for i, task in enumerate(task_windows):
        nodes.append(i)
        if np.isnan(task[1]):
            window_starts.append(0)
            window_durations.append(endval)
        else:
            window_starts.append(task[0])
            window_durations.append(task[1] - task[0])
        if visited[i] > 0.5:
            visited_nodes.append(i)
            window_service_starts.append(visit_times[i])
            window_service_durations.append(task[2])

    ax2.set_title('Time Schedule')
    ax2.barh(nodes, window_durations, left=window_starts, color='w', edgecolor='b')

    visit_order = np.arange(num_nodes)
    visit_times = visit_times[visited > 0.5]
    visit_order = visit_order[visited > 0.5]
    idxs = np.argsort(visit_times)
    visit_order = visit_order[idxs]
    visit_times = visit_times[idxs]

    ax2.barh(visited_nodes, window_service_durations, left=window_service_starts, color='g')

    visit_order = np.append(visit_order, 0)
    visit_times = np.append(visit_times, total_cost - wait_times[visit_order[-1]])
    visit_times_waits = []
    visit_order_waits = []
    for i, val in enumerate(visit_order):
        visit_order_waits.append(val)
        visit_order_waits.append(val)  # twice because of wait
        visit_times_waits.append(visit_times[i])
        visit_times_waits.append(visit_times[i] + wait_times[val])

    del visit_times_waits[-2]
    del visit_order_waits[-2]
    ax2.plot(visit_times_waits, visit_order_waits, 'ro-')
    plt.show()


def get_solution(node_rewards, time_windows, cost_matrix, max_cost=0):
    num_nodes = node_rewards.shape[0]
    x = c.Variable(cost_matrix.shape, boolean=True)

    # variables in subtour elimination constraints
    s = c.Variable(node_rewards.shape, nonneg=True)

    # cost = c.trace(cost_matrix @ x) + c.sum(s)  # total cost of the tour
    mask_vec = np.zeros(num_nodes)
    mask_vec[0] = 1.0
    cost = c.max(s) + mask_vec @ cost_matrix @ x[:, 0]
    profit = c.sum(x @ node_rewards)
    y = x @ np.ones((num_nodes, 1))

    constraints = []

    constraints.append(cost <= max_cost)

    # we leave from the first node
    constraints.append(c.sum(x[0, :]) == 1)
    # we come back to the first node
    constraints.append(c.sum(x[:, 0]) == 1)

    # max one connection outgoing and incoming
    ones_arr = np.ones((node_rewards.shape[0], 1))  # array for ones
    # max one connection outgoing and incoming
    constraints.append(x.T @ ones_arr <= 1)
    constraints.append(x @ ones_arr <= 1)

    for i in range(num_nodes):
        constraints.append(c.sum(x[:, i]) == c.sum(x[i, :]))

    M = 5000.0

    # constraints.append(s[0] == 0)
    for i in range(num_nodes):
        # constraints.append(time_windows[i, 0] - s[i] - M * (1 - y[i]) <= 0)
        constraints.append(time_windows[i, 0] - s[i] + M * c.sum(x[:, i]) <= M)
        # constraints.append(s[i] + time_windows[i, 2] - time_windows[i, 1] - M * (1 - y[i]) <= 0)
        if not np.isnan(time_windows[i, 1]):
            constraints.append(s[i] + time_windows[i, 2] - time_windows[i, 1] + M * c.sum(x[:, i]) <= M)
        for j in range(num_nodes):
            if i == j:
                constraints.append(x[i, j] == 0)
            elif j == 0:
                pass
            else:
                constraints.append(s[i] + time_windows[i, 2] + cost_matrix[i, j] - s[j] + M * x[i, j] <= M)

    prob = c.Problem(c.Maximize(profit), constraints)

    prob.solve()

    if np.any(x.value is None):  # no feasible solution found!

        print('Feasible solution not found, lower your time constraint!')
        # print(x.value)
        raise ValueError()

    print('----- Visited -----')
    print(np.around(y.value, 2))
    print('----- S -----')
    print(np.around(s.value * y.value.T, 2))

    return x.value, s.value, prob.value, cost.value, prob.solver_stats.solve_time


if __name__ == '__main__':
    np.random.seed(1)

    files = [os.path.join('.', 'Maps', f) for f in os.listdir('Maps') if f[-3:] == 'mat' and f[-12:-4] != 'solution']
    print('Files Found: {}'.format(files))
    for f in files:
        cost_matrix = load_cost_matrix(f)
        cost_matrix = cost_matrix[0:5, 0:5]
        # print('----- Edge Costs -----')
        # print(cost_matrix)
        num_nodes = cost_matrix.shape[0]

        # this will generate a random score matrix.
        score_vector = np.ones(num_nodes)
        # score_vector[0] = 0
        task_windows = setup_task_windows(score_vector)
        print(np.around(task_windows, 2).astype('float'))

        max_cost = get_starting_cost(cost_matrix, task_windows)
        try:
            plan, visit_times, profit, cost, solve_time = get_solution(score_vector, task_windows, cost_matrix, max_cost)
        except ValueError:
            continue
        wait_times = compute_wait_times(plan, cost_matrix, task_windows, visit_times, cost)
        visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, cost)
        save_solution(f, visit_order_waits, visit_times_waits, solver_type='tw_st')

        print('----- Plan -----')
        print(np.around(plan, 2).astype('int32'))
        print('----- Edge Costs -----')
        print(cost_matrix)
        print('----- Wait Times -----')
        print(np.around(wait_times, 2))

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

        display_results(g, plan, task_windows, visit_times, wait_times, cost)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cwb
"""

# Orienteering problem with Miller-Tucker-Zemlin formulation
# Service window formulation added by cwb

import cvxpy as c
import numpy as np
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

        next_node = np.argmax(path_solution[now_node, :])  # where we go from node i
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


def compute_wait_times(plan, edge_costs, visit_times, final_cost):
    # wait time at 0 = visit_times[state_2] - edge_cost[0, state_2]
    # state_2 = argmax(plan[0,:])
    wait_times = np.zeros(visit_times.shape)
    curr_state = 0
    for i in range(len(visit_times)):
        next_state = np.argmax(plan[curr_state, :])
        cost = edge_costs[curr_state, next_state]
        if next_state == 0:
            wait_times[curr_state] = final_cost - cost - visit_times[curr_state]
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
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, width=20,
                                 edge_color='b', ax=ax1)
    # plt.savefig('{}.png'.format(save_name))
    nodes = []
    windows = []
    offsets = []
    fake_val = task_windows[0][1]  # the start node always has an "open" window
    for i, task in enumerate(task_windows):
        nodes.append(i)
        if task[1] == fake_val:
            offsets.append(0)
            windows.append(total_cost)
        else:
            offsets.append(task[0])
            windows.append(task[1] - task[0])

    ax2.set_title('Time Schedule')
    ax2.barh(nodes, windows, left=offsets)
    visited = np.reshape(visited, -1)
    visit_order = np.arange(num_nodes)
    visit_times = visit_times[visited > 0.5]
    visit_order = visit_order[visited > 0.5]
    idxs = np.argsort(visit_times)
    visit_order = visit_order[idxs]
    visit_times = visit_times[idxs]
    visit_order = np.append(visit_order, 0)
    visit_times = np.append(visit_times, total_cost - wait_times[visit_order[-1]])
    visit_times_waits = []
    visit_order_waits = []
    for i, val in enumerate(visit_order):
        visit_order_waits.append(val)
        visit_order_waits.append(val)  # twice because of wait
        visit_times_waits.append(visit_times[i])
        visit_times_waits.append(visit_times[i] + wait_times[val])

    ax2.plot(visit_times_waits, visit_order_waits, 'ro-')
    plt.show()


def setup_task_windows(score_vector):
    windows = np.zeros((score_vector.shape[0], 2))
    curr_time = np.random.uniform(20, 28)
    for row in range(score_vector.shape[0]):
        if row % 2 == 1:
            task_duration = np.random.uniform(10, 20)
            windows[row, :] = (curr_time, curr_time + task_duration)
            curr_time += task_duration
            curr_time += np.random.uniform(10, 38)
        else:
            # set half the time windows to (-inf, inf), same as no window
            windows[row, :] = (0.0, 10000.0)
    return windows


def get_solution(node_rewards, time_windows, cost_matrix):
    num_nodes = node_rewards.shape[0]
    x = c.Variable(cost_matrix.shape, boolean=True)

    # variables in subtour elimination constraints
    s = c.Variable(node_rewards.shape)

    ones_arr = np.ones(node_rewards.shape)  # array for ones

    # cost = c.trace(cost_matrix @ x) + c.sum(s)  # total cost of the tour
    mask_vec = np.zeros(num_nodes)
    mask_vec[0] = 1.0
    cost = c.max(s) + mask_vec @ cost_matrix @ x[:, 0]
    profit = c.sum(x @ node_rewards)
    y = x @ np.ones((num_nodes, 1))

    constraints = []

    # we leave from the first node
    constraints.append(c.sum(x[0, :]) == 1)
    # we come back to the first node
    constraints.append(c.sum(x[:, 0]) == 1)

    # max one connection outgoing and incoming
    constraints.append(x.T @ ones_arr <= 1)
    constraints.append(x @ ones_arr <= 1)

    for i in range(num_nodes):
        constraints.append(c.sum(x[:, i]) == c.sum(x[i, :]))

    # let us add the time constraints
    total_time = np.max(time_windows[time_windows[:, 1] < np.max(time_windows[:, 1]), 1])
    fake_edge_val = np.max(cost_matrix)
    total_time += 2 * np.max(cost_matrix[cost_matrix < fake_edge_val])
    total_time = np.round(total_time, 0)
    print('Time Limit: {}'.format(total_time))
    constraints.append(cost <= total_time)

    M = 10000.0

    constraints.append(s[0] == 0)
    for i in range(num_nodes):
        constraints.append(time_windows[i, 0] - s[i] - M * (1 - y[i]) <= 0)
        constraints.append(s[i] - time_windows[i, 1] - M * (1 - y[i]) <= 0)
        for j in range(1, num_nodes):
            constraints.append(s[i] + cost_matrix[i, j] - s[j] - M * (1 - x[i, j]) <= 0)

    prob = c.Problem(c.Maximize(profit), constraints)

    prob.solve()

    if np.any(x.value is None):  # no feasible solution found!

        print('Feasible solution not found, lower your time constraint!')
        print(x.value)
        raise ValueError()

    print('----- Visited -----')
    print(np.around(y.value, 2))
    print('----- S -----')
    print(np.around(s.value * y.value.T, 2))

    return x.value, s.value, prob.value, cost.value, prob.solver_stats.solve_time


if __name__ == '__main__':
    np.random.seed(1)

    # number of nodes in the Orienteering
    num_nodes = 8
    # the time horizons which we try out
    try_times = range(20, 100, 20)
    try_times = [20]

    for total_time in try_times:
        # this will generate a random cost matrix.
        cost_matrix = np.random.randint(1, 15, (num_nodes, num_nodes))
        # ensure symmetry of the matrix
        cost_matrix = cost_matrix + cost_matrix.T
        # make sure we don't travel from node to same node, by having high cost.
        np.fill_diagonal(cost_matrix, 1000)

        # this will generate a random score matrix.
        score_vector = np.random.uniform(1, 5, num_nodes)
        score_vector[0] = 0
        task_windows = setup_task_windows(score_vector)
        print(np.around(task_windows, 2).astype('float'))
        # since the 0th node, start node, has no value!

        plan, visit_times, profit, cost, solve_time = get_solution(score_vector, task_windows, cost_matrix)
        wait_times = compute_wait_times(plan, cost_matrix, visit_times, cost)

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

    # print(cost_matrix)
    # print(x.value)
    # print(score_vector)

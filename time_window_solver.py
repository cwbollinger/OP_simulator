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

from utils import load_cost_matrix, save_solution, build_graph, get_arrive_depart_pairs, setup_task_windows, get_starting_cost, get_constraints, get_plan_score

from timeit import default_timer as timer


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
        nodeval[k] = (k + 1, np.around(nodeval[k], 2))
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

    visit_order_waits = [x + 1 for x in visit_order_waits]
    ax2.plot(visit_times_waits, visit_order_waits, 'ro-')
    plt.show()


def get_solution(node_rewards, cost_matrix, time_windows, time_budget=0):
    x = c.Variable(cost_matrix.shape, boolean=True)

    # variables in subtour elimination constraints
    s = c.Variable(node_rewards.shape, nonneg=True)

    constraints = get_constraints(cost_matrix, node_rewards, x, s,
                                  time_windows=time_windows)

    cost = c.max(s) + c.sum(c.multiply(cost_matrix, x)[:, 0])
    constraints.append(cost <= time_budget)

    reward = c.sum(x @ node_rewards)
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

    return x.value, s.value, prob.value, cost.value, end - start


def compute_success_rate():
    files = [os.path.join('.', 'Maps', f) for f in os.listdir('Maps')
             if f[-3:] == 'mat' and f[-12:-4] != 'solution']
    print('Files Found: {}'.format(files))
    success_count = {}
    for window_ratio in [1, 2, 3, 4, 5]:
        success_count[window_ratio] = 10
        for f in files:
            cost_matrix = load_cost_matrix(f)
            num_nodes = cost_matrix.shape[0]

            score_vector = np.ones(num_nodes)
            task_windows = setup_task_windows(score_vector, window_ratio)

            max_cost = get_starting_cost(cost_matrix, task_windows)
            try:
                plan, visit_times, profit, cost, solver_stats = get_solution(score_vector, cost_matrix, task_windows, max_cost)
            except ValueError:
                print('Failed with cost {}'.format(max_cost))
                success_count[window_ratio] -= 1
                continue

            wait_times = compute_wait_times(plan, cost_matrix, visit_times)

            visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, cost)
            save_solution(f, visit_order_waits, visit_times_waits, solver_type='tw')

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
        success_count[window_ratio] /= 10.0

    print('Success probabilities across constraint ratios')
    print(success_count)
    return success_count


def compute_solve_times(files):
    problem_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    small_map_solve_times = {p: [] for p in problem_sizes}
    large_map_solve_times = {p: [] for p in problem_sizes}
    for n in problem_sizes:
        for f in files:
            cost_matrix = load_cost_matrix(f)
            # cost_matrix = cost_matrix[0:n, 0:n]
            # print('----- Edge Costs -----')
            # print(cost_matrix)
            num_nodes = cost_matrix.shape[0]

            score_vector = np.ones(num_nodes)
            task_windows = setup_task_windows(score_vector)

            max_cost = get_starting_cost(cost_matrix, task_windows)
            try:
                plan, visit_times, profit, cost, solver_stats = get_solution(score_vector, cost_matrix, task_windows, max_cost)
            except ValueError:
                print('Failed with cost {}'.format(max_cost))
                continue

            wait_times = compute_wait_times(plan, cost_matrix, visit_times)

            visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, cost)
            save_solution(f, visit_order_waits, visit_times_waits, solver_type='tw')

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
            msg = 'Time taken: {} seconds'
            time = solver_stats.solve_time + solver_stats.setup_time
            print(msg.format(time))
            # display_results(g, plan, task_windows, visit_times, wait_times, cost)
            if f[-12:-7] == '20x20':
                small_map_solve_times[n].append(time)
            elif f[-12:-7] == '50x50':
                large_map_solve_times[n].append(time)

    return (small_map_solve_times, large_map_solve_times)


def get_time_data():
    np.random.seed(1)

    files = [os.path.join('.', 'Maps', f) for f in os.listdir('Maps')
             if f[-3:] == 'mat' and f[-12:-4] != 'solution']
    maps20x20 = [f for f in files if '20x20' in f]
    maps50x50 = [f for f in files if '20x20' in f]
    big_maps = [f for f in files if '100_POI' in f]

    runtimes = {}
    for n in [4, 6, 8, 10, 12, 14]:
        print(n)
        runtimes[n] = []
        for f in big_maps:
            cost_matrix = load_cost_matrix(f)
            # high diagonal costs cause numerical errors
            # use constraints to prevent travel to self
            # diagonal cost must be > 0 for this to work
            # but should be low
            np.fill_diagonal(cost_matrix, 1)
            cost_matrix = cost_matrix[0:n, 0:n]
            # print('----- Edge Costs -----')
            # print(cost_matrix)
            num_nodes = cost_matrix.shape[0]

            score_vector = np.ones(num_nodes)
            task_windows = setup_task_windows(score_vector)
            # print('----- Task Windows -----')
            # print(np.around(task_windows, 2).astype('float'))
            max_cost = get_starting_cost(cost_matrix, task_windows)

            plan, visit_times, profit, cost, solve_time = get_solution(score_vector, cost_matrix, task_windows, max_cost)
            runtimes[n].append(solve_time)

            wait_times = compute_wait_times(plan, cost_matrix, visit_times)

            visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, cost)
            save_solution(f, visit_order_waits, visit_times_waits, solver_type='tw')

            print('----- Visited -----')
            print(np.sum(plan, axis=1))
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
            msg = 'Time taken: {} seconds'
            print(msg.format(solve_time))
            # display_results(g, plan, task_windows, visit_times, wait_times, cost)

    with open('results_tw.txt', 'w') as f:
        f.write(str(runtimes))


if __name__ == '__main__':
    np.random.seed(1)
    files = [os.path.join('.', 'Maps', f) for f in os.listdir('Maps')
             if f[-3:] == 'mat' and f[-12:-4] != 'solution' and 'Q' not in f]
    maps20x20 = [f for f in files if '20x20' in f]

    rewards = {}
    times = {}
    # for budget in [50, 100, 150, 200, 250, 300]:
    for budget in [600, 500, 400, 300, 200]:
        print('Budget: {}'.format(budget))
        rewards[budget] = []
        times[budget] = []
        for f in maps20x20:
            cost_matrix = load_cost_matrix(f)
            np.fill_diagonal(cost_matrix, 1)
            num_nodes = cost_matrix.shape[0]

            score_vector = np.ones(num_nodes)
            task_windows = setup_task_windows(score_vector)

            try:
                plan, visit_times, profit, cost, solve_time = get_solution(score_vector, cost_matrix, task_windows, budget)
            except ValueError:
                print('No solution for {}'.format(f))
                continue

            wait_times = compute_wait_times(plan, cost_matrix, visit_times)

            visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, cost)
            save_solution(f, visit_order_waits, visit_times_waits, solver_type='tw')

            # print('----- Visited -----')
            # print(np.sum(plan, axis=1))
            # print('----- Plan -----')
            # print(np.around(plan, 2).astype('int32'))
            # print('----- Edge Costs -----')
            # print(cost_matrix)
            # print('----- Wait Times -----')
            # print(np.around(wait_times, 2))

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

            score = get_plan_score(task_windows, plan, visit_times, task_windows[:, 2])
            rewards[budget].append(score)
            times[budget].append(solve_time)
            print(score)

            msg = 'Profit: {:.2f}, cost: {:.2f}, verification cost: {:.2f}'
            print(msg.format(profit, cost, verified_cost))
            msg = 'Time taken: {} seconds'
            print(msg.format(solve_time))
            # display_results(g, plan, task_windows, visit_times, wait_times, cost)

    print(rewards)
    with open('rewards_tw.txt', 'w') as f:
        f.write(str(rewards))
    with open('times_tw.txt', 'w') as f:
        f.write(str(times))

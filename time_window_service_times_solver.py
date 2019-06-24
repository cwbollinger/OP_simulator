#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cwb
"""

# Orienteering problem with Miller-Tucker-Zemlin formulation
# Service window formulation added by cwb

import os
import sys

import cvxpy as c
import numpy as np
import networkx as nx

from utils import load_cost_matrix, save_solution, build_graph, get_arrive_depart_pairs, setup_task_windows, get_starting_cost, get_constraints, get_plan_score

from timeit import default_timer as timer


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
    import matplotlib.pyplot as plt
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


def get_solution(node_rewards, cost_matrix, time_windows, max_cost=0):
    x = c.Variable(cost_matrix.shape, boolean=True)

    # variables in subtour elimination constraints
    s = c.Variable(node_rewards.shape, nonneg=True)

    service_times = time_windows[:, 2]
    constraints = get_constraints(cost_matrix, node_rewards, x, s,
                                  time_windows=time_windows[:, 0:2],
                                  service_times=service_times)

    cost = c.max(s) + c.sum(c.multiply(cost_matrix, x)[:, 0])
    constraints.append(cost <= max_cost)

    profit = c.sum(x @ node_rewards)
    prob = c.Problem(c.Maximize(profit), constraints)

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


def get_time_data():
    np.random.seed(1)

    files = [os.path.join('.', 'Maps', f) for f in os.listdir('Maps')
             if f[-3:] == 'mat' and f[-12:-4] != 'solution']
    maps20x20 = [f for f in files if '20x20' in f]
    maps50x50 = [f for f in files if '20x20' in f]
    big_maps = [f for f in files if '100_POI' in f]
    runtimes = {}
    for n in [4, 6, 8, 10, 12, 14]:
        runtimes[n] = []
        for f in big_maps[:10]:
            print(f)
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

            # this will generate a random score matrix.
            score_vector = np.ones(num_nodes)
            # score_vector[0] = 0
            task_windows = setup_task_windows(score_vector)
            # print(np.around(task_windows, 2).astype('float'))

            max_cost = get_starting_cost(cost_matrix, task_windows)
            plan, visit_times, profit, cost, solve_time = get_solution(score_vector, cost_matrix, task_windows, max_cost)
            runtimes[n].append(solve_time)

            wait_times = compute_wait_times(plan, cost_matrix, task_windows, visit_times, cost)
            visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, cost)
            save_solution(f, visit_order_waits, visit_times_waits, solver_type='tw_st')

            # print('----- Plan -----')
            # print(np.around(plan, 2).astype('int32'))
            # print('----- Edge Costs -----')
            # print(cost_matrix)
            # print('----- Wait Times -----')
            # print(np.around(wait_times, 2))

            g, tour, verified_cost = build_graph(plan, score_vector, cost_matrix)

            msg = 'The maximum profit tour found is \n'
            for idx, k in enumerate(tour):
                msg += str(k)
                if idx < len(tour) - 1:
                    msg += ' -> '
                else:
                    msg += ' -> 0'
            print(msg)

            print(profit)
            print(cost)
            print('Profit: {:.2f}, cost: {:.2f}, verification cost: {:.2f} '.format(profit, cost, verified_cost))
            print('Time taken: {:.2f} seconds'.format(solve_time))

            # display_results(g, plan, task_windows, visit_times, wait_times, cost)

    with open('results_twst.txt', 'w') as f:
        f.write(str(runtimes))


def get_budget_performance():
    np.random.seed(1)
    files = [os.path.join('.', 'Maps', f) for f in os.listdir('Maps')
             if f[-3:] == 'mat' and f[-12:-4] != 'solution' and 'Q' not in f]
    maps20x20 = [f for f in files if '20x20' in f]

    rewards = {}
    times = {}
    for budget in [600, 500, 400, 300, 200]:
        print('Budget: {}'.format(budget))
        rewards[budget] = []
        times[budget] = []
        for f in maps20x20:
            print(f)
            cost_matrix = load_cost_matrix(f)
            np.fill_diagonal(cost_matrix, 1)
            # cost_matrix = cost_matrix[0:n, 0:n]
            # print('----- Edge Costs -----')
            # print(cost_matrix)
            num_nodes = cost_matrix.shape[0]

            # this will generate a random score matrix.
            score_vector = np.ones(num_nodes)
            task_windows = setup_task_windows(score_vector)

            # max_cost = get_starting_cost(cost_matrix, task_windows)
            try:
                plan, visit_times, profit, cost, solve_time = get_solution(score_vector, cost_matrix, task_windows, budget)
            except ValueError:
                print('No solution for {}'.format(f))
                continue

            wait_times = compute_wait_times(plan, cost_matrix, task_windows, visit_times, cost)
            visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, cost)
            save_solution(f, visit_order_waits, visit_times_waits, solver_type='tw_st')

            g, tour, verified_cost = build_graph(plan, score_vector, cost_matrix)

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
            print('True Score: {}'.format(score))

            print(profit)
            print(cost)
            print('Profit: {:.2f}, cost: {:.2f}, verification cost: {:.2f} '.format(profit, cost, verified_cost))
            print('Time taken: {:.2f} seconds'.format(solve_time))

            # display_results(g, plan, task_windows, visit_times, wait_times, cost)

    with open('rewards_twst.txt', 'w') as f:
        f.write(str(rewards))
    with open('times_twst.txt', 'w') as f:
        f.write(str(times))


def load_problem():
    costs = np.load(os.path.abspath('./distances.npy'))
    tws = np.load(os.path.abspath('./windows.npy'))
    return costs, tws


def save_problem(visit_order, visit_times):
    solution = np.zeros((len(visit_order), 2))
    for i in range(len(visit_order)):
        solution[i, 0] = visit_order[i]
        solution[i, 1] = visit_times[i]

    np.save(os.path.abspath('./solution.npy'), solution)


if __name__ == '__main__':
    # Make sure our cwd is the directory where this file is located
    print(sys.argv[0])
    script_dir = os.path.dirname(sys.argv[0])
    os.chdir(os.path.abspath(script_dir))
    # load cost_matrix
    cost_matrix, task_windows = load_problem()
    num_nodes = cost_matrix.shape[0]

    if num_nodes == 0:
        print('What, no nodes???')
        sys.exit(-1)

    # this will generate a random score matrix.
    score_vector = np.ones(num_nodes)
    # task_windows = setup_task_windows(score_vector)

    # max_cost = get_starting_cost(cost_matrix, task_windows)
    # fake budget for now... (end of last service window + 1 Hour)
    budget = np.max(task_windows[:, 1]) + 3600.0
    try:
        plan, visit_times, profit, cost, solve_time = get_solution(score_vector, cost_matrix, task_windows, budget)
    except ValueError:
        print('No solution found!')

    wait_times = compute_wait_times(plan, cost_matrix, task_windows, visit_times, cost)
    visit_order_waits, visit_times_waits = get_arrive_depart_pairs(plan, visit_times, wait_times, cost)
    save_problem(visit_order_waits, visit_times_waits)

    # g, tour, verified_cost = build_graph(plan, score_vector, cost_matrix)

    # msg = 'The maximum profit tour found is \n'
    # for idx, k in enumerate(tour):
    #     msg += str(k)
    #     if idx < len(tour) - 1:
    #         msg += ' -> '
    #     else:
    #         msg += ' -> 0'
    # print(msg)

    # score = get_plan_score(task_windows, plan, visit_times, task_windows[:, 2])

import numpy as np
import scipy.io as sio
import networkx as nx
import cvxpy as c


def get_plan_score(task_windows, plan, arrival_times, service_times=None):
    curr_node = 0
    score = 0
    while True:
        window = np.around(task_windows[curr_node, 0:2], 2)
        # print(window)
        a_time = np.around(arrival_times[curr_node], 2)
        arrive_after = window[0] <= a_time

        s_time = 0 if service_times is None else np.around(service_times[curr_node], 2)
        if np.isnan(window[1]):
            depart_before = True
        else:
            depart_before = a_time + s_time <= window[1]

        if arrive_after and depart_before:
            score += 1
        else:
            print('{}:'.format(curr_node, arrive_after, depart_before))
            if not arrive_after:
                print('False: {} <= {}'.format(window[0], a_time))
            if not depart_before:
                print('False {} + {} <= {}'.format(a_time, s_time, window[1]))
        curr_node = np.argmax(plan[curr_node, :])
        if curr_node == 0:
            break
    return score


def get_starting_cost(cost_matrix, time_windows):
    total_time = np.max(time_windows[~np.isnan(time_windows[:, 1]), 1])
    fake_edge_val = np.max(cost_matrix)
    total_time += 15 * np.max(cost_matrix[cost_matrix < fake_edge_val])
    return total_time


def load_cost_matrix(filename):
    mat = sio.loadmat(filename)
    return mat['A'][0][0][2]


def get_random_cost_matrix(num_nodes=10):
    cost_matrix = np.random.randint(1, 15, (num_nodes, num_nodes))
    # ensure symmetry of the matrix
    cost_matrix = cost_matrix + cost_matrix.T
    # make sure we don't travel from node to same node.
    np.fill_diagonal(cost_matrix, 111111)


def save_solution(filename, visit_order, visit_times, solver_type=''):
    solution = np.zeros((len(visit_order), 2))
    for i in range(len(visit_order)):
        solution[i, 0] = visit_order[i]
        solution[i, 1] = visit_times[i]
    # print(np.around(solution, 2))
    filename = filename[:-4]
    if solver_type != '':
        filename += '_' + solver_type
    filename += '_solution.mat'
    sio.savemat(filename, {'problem_solution': solution})


def get_arrive_depart_pairs(plan, visit_times, wait_times, total_cost):
    num_nodes = plan.shape[0]
    visited = plan @ np.ones((num_nodes, 1))
    visited = np.reshape(visited, -1)
    visit_order = np.arange(num_nodes)
    visit_times = visit_times[visited > 0.5]
    visit_order = visit_order[visited > 0.5]
    idxs = np.argsort(visit_times)
    visit_order = visit_order[idxs]
    visit_times = visit_times[idxs]
    visit_order = np.append(visit_order, 0)
    # visit_times = np.append(visit_times, total_cost - wait_times[visit_order[-1]])
    visit_times = np.append(visit_times, total_cost)
    visit_times_waits = []
    visit_order_waits = []
    for i, val in enumerate(visit_order):
        visit_order_waits.append(val)
        visit_order_waits.append(val)  # twice because of wait
        visit_times_waits.append(visit_times[i])
        if i != 0 and val == 0:
            visit_times_waits.append(visit_times[i])
        else:
            visit_times_waits.append(visit_times[i] + wait_times[val])

    return visit_order_waits, visit_times_waits


def build_graph(path_solution, node_scores, edge_costs):
    g = nx.DiGraph()

    num_nodes = path_solution.shape[0]

    verified_cost = 0
    now_node = 0  # Initialize at start node
    tour = []

    # g.add_nodes_from(range(1,num_nodes+1))
    for k in range(edge_costs.shape[0]):
        g.add_node(k, value=node_scores[k])  # 1 based indexing

    counter = 0
    while(True):  # till we reach end node
        if counter > num_nodes:
            print('Something has gone horribly wrong')
            break
        counter += 1
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


def setup_task_windows(score_vector, constrained_ratio=1):
    # ratio is free:constrained
    # only whole ratios supported for now
    windows = np.zeros((score_vector.shape[0], 3))
    curr_time = np.random.uniform(20, 38)
    for row in range(score_vector.shape[0]):
        if row % (constrained_ratio + 1) == 1:
            task_window_size = np.round(np.random.uniform(25, 40), 2)
            # how much of the window is needed to complete the task?
            task_duration = np.random.uniform(0.33, 0.75) * task_window_size
            windows[row, :] = (curr_time, curr_time + task_window_size, task_duration)
            curr_time += np.round(task_window_size, 2)
            curr_time += np.round(np.random.uniform(65, 95), 2)
        else:
            task_duration = np.round(np.random.uniform(2, 14), 2)
            windows[row, :] = (0.0, None, task_duration)
    return windows


def old_get_constraints(costs, rewards, x, u, time_windows=None, service_times=False):
    num_nodes = costs.shape[0]
    constraints = []

    # we leave from the first node
    constraints.append(c.sum(x[0, 1:]) == 1)
    # we come back to the first node
    constraints.append(c.sum(x[1:, 0]) == 1)

    ones_arr = np.ones(rewards.shape)  # array for ones
    # max one connection outgoing and incoming
    constraints.append(x @ ones_arr <= 1)
    constraints.append(x.T @ ones_arr <= 1)

    for k in range(1, num_nodes):
        constraints.append(c.sum(x[:, k]) == c.sum(x[k, :]))

    # subtour elimination constraints (Miller-Tucker-Zemlin similar formulation)
    constraints.append(0 <= u)
    constraints.append(u <= num_nodes)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j != 0:
                constraints.append(u[i] + 1 - u[j] <= num_nodes * (1 - x[i, j]))
    return constraints


def get_constraints(costs, rewards, x, s, time_windows=None, service_times=None):
    num_nodes = costs.shape[0]
    constraints = []

    if time_windows is not None:
        tt = np.copy(costs).astype('float')
        limit = 5000.0
    else:
        tt = np.ones((num_nodes, num_nodes))
        limit = num_nodes

    # we leave from the first node
    constraints.append(c.sum(x[0, 1:]) == 1)
    # we come back to the first node
    constraints.append(c.sum(x[1:, 0]) == 1)

    y = c.sum(x, axis=1)

    ones_arr = np.ones(rewards.shape)  # array for ones
    # max one connection outgoing and incoming
    constraints.append(x @ ones_arr <= 1)
    constraints.append(x.T @ ones_arr <= 1)

    for k in range(1, num_nodes):
        constraints.append(c.sum(x[:, k]) == c.sum(x[k, :]))

        # only include time window constraint if it exists for this node
        if time_windows is not None and not np.isnan(time_windows[k, 1]):
            constraints.append(time_windows[k, 0] - s[k] <= limit * (1 - y[k]))
            st = 0 if service_times is None else service_times[k]
            constraints.append(s[k] + st - time_windows[k, 1] <= limit * (1 - y[k]))

    constraints.append(0 <= s)
    constraints.append(s <= limit)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j != 0:
                constraints.append(s[i] + tt[i, j] - s[j] <= limit * (1 - x[i, j]))
    return constraints


def print_constraints_solution(costs, rewards, good_x, good_u, budget):
        maybe_valid = get_constraints(costs, rewards, good_x, good_u)
        cost = c.sum(c.multiply(costs, good_x))  # total cost of the tour
        maybe_valid.append(cost <= budget)
        print(' ----- x -----')
        print(good_x)
        print(' ----- u -----')
        print(good_u)
        print('-----')
        for i, x in enumerate(maybe_valid):
            print('{}: '.format(i), end='')
            if hasattr(x, 'value'):
                print(x.value())
            else:
                print(x)
        print('-----')

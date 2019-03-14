import numpy as np
import scipy.io as sio
import networkx as nx


def get_starting_cost(cost_matrix, time_windows):
    total_time = np.max(time_windows[~np.isnan(time_windows[:, 1]), 1])
    fake_edge_val = np.max(cost_matrix)
    total_time += 30 * np.max(cost_matrix[cost_matrix < fake_edge_val])
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
    print(np.around(solution, 2))
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
            curr_time += np.round(np.random.uniform(55, 95), 2)
        else:
            task_duration = np.round(np.random.uniform(2, 14), 2)
            windows[row, :] = (0.0, None, task_duration)
    return windows

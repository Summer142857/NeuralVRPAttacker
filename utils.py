import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import random
import time, pytz
from datetime import datetime
import logging
import shutil


import elkai
import pandas as pd

import torch
from scipy.spatial.distance import jensenshannon as js


process_start_time = datetime.now(pytz.timezone("Europe/Amsterdam"))
result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_result_folder():
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename) else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].extend(value)
        else:
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                value = list(args)
            elif type(args) == list:
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        return self.keys


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total - count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time * 60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time * 60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def util_print_log_array(logger, result_log: LogData):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key + '_list', result_log.get(key)))


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def compute_euclidean_distance_matrix(locations):
    num_nodes = locations.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))

    # Calculate the Euclidean distance between each pair of nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i, j] = euclidean_distance(locations[i], locations[j])
    return distance_matrix


def calculate_total_distance(B, A):
    """
    B: tour
    A: distance matrix
    """
    total_distance = 0
    for i in range(len(B) - 1):
        from_node = B[i]
        to_node = B[i + 1]
        total_distance += A[from_node][to_node]

    # Add the distance from the last node back to the starting node to complete the tour
    total_distance += A[B[-1]][B[0]]

    return total_distance


def lkh_atsp(problem):
    # problem is in the adj matrix form
    if isinstance(problem, torch.Tensor):
        problem = problem.detach().cpu().numpy()
    costs = []
    for i in range(problem.shape[0]):
        matrix = problem[i, :, :]*100000
        cities = elkai.DistanceMatrix(matrix.tolist())
        tour = cities.solve_tsp()
        cost = calculate_total_distance(tour, matrix/100000)
        costs.append(cost)
    return costs


def lkh(problem):
    if isinstance(problem, torch.Tensor):
        problem = problem.detach().cpu().numpy()
    costs = []
    for i in range(problem.shape[0]):
        coords = problem[i, :, :]*1000
        distances = compute_euclidean_distance_matrix(coords)
        cities = elkai.DistanceMatrix(distances)
        tour = cities.solve_tsp()
        cost = calculate_total_distance(tour, distances/1000)
        costs.append(cost)
    return costs

def resample_nodes(problem, sample_num = 50, type = "TSP"):
    if type == "TSP":
        nodes_num = problem.shape[1]
        random_indices = torch.randperm(nodes_num)[:sample_num]
        sampled_tensor = problem[:, random_indices, :]
        return sampled_tensor
    elif type == "CVRP":
        depot_xy = problem[0]
        nodes_num = problem[1].shape[1]
        random_indices = torch.randperm(nodes_num)[:sample_num]
        sampled_nodes = problem[1][:, random_indices, :]
        sampled_demand = problem[2][:, random_indices]
        return (depot_xy, sampled_nodes, sampled_demand)
    else:
        assert 0, "problem type not implemented"

def draw_batch_instances(gap, new_problems):
    batch = gap.shape[0]
    if batch == 64:
        idx = torch.tensor([i[-1] for i in gap.sort(dim=1)[1]])
        pp = new_problems.gather(1, idx[:, None, None, None].expand(batch, 1, 50, 2)).squeeze(1).cpu().numpy()
        fig, ax = plt.subplots(8, 8, figsize=(16, 16), sharex=True, sharey=True)
        for i in range(64):
            row = i // 8
            col = i % 8
            ax[row, col].scatter(pp[i, :, 0], pp[i, :, 1])
        plt.tight_layout()
        plt.show()
    elif batch == 32:
        idx = torch.tensor([i[-1] for i in gap.sort(dim=1)[1]])
        pp = new_problems.gather(1, idx[:, None, None, None].expand(batch, 1, 100, 2)).squeeze(1).cpu().numpy()
        fig, ax = plt.subplots(4, 8, figsize=(8, 16), sharex=True, sharey=True)
        for i in range(32):
            row = i // 8
            col = i % 4
            ax[row, col].scatter(pp[i, :, 0], pp[i, :, 1])
        plt.tight_layout()
        plt.show()
    else:
        pass

def cal_js(q, p):
    # shape: (batch, head_num, n, problem)
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    res = js(q, p)
    return res

def get_diverse_metric(gap, new_problems, bins):
    js_divergences = []
    pseudo_count = 1e-6

    batch_size = gap.shape[0]
    idx = torch.tensor([i[-1] for i in gap.sort(dim=1)[1]])
    most_successful_instances = new_problems.gather(1, idx[:, None, None, None].expand(batch_size, 1, 50, 2)).squeeze(1)
    for i in range(most_successful_instances.shape[0]-1):
        for j in range(i+1, most_successful_instances.shape[0]):
            # Compute the empirical probability distributions with smoothing
            prob_dist_A = torch.torch.histogramdd(most_successful_instances[i].cpu(), bins=[bins, bins], density=True)[0].view(-1) + pseudo_count
            prob_dist_B = torch.torch.histogramdd(most_successful_instances[j].cpu(), bins=[bins, bins], density=True)[0].view(-1) + pseudo_count

            js_divergence = cal_js(prob_dist_A, prob_dist_B)

            js_divergences.append(js_divergence)
    return torch.tensor(js_divergences)

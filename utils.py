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


def hist2d_cities():
    import scienceplots
    plt.style.use(['science', 'no-latex'])

    data = torch.load('./TSP/POMO.pt')

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    flat_data = data.reshape(-1, 2)
    # Extract x and y coordinates
    x = flat_data[:, 0]
    y = flat_data[:, 1]

    plt.rcParams.update({'font.size': 15, 'font.weight': 'bold'})

    # Create a 2D histogram
    plt.figure(figsize=(4.5, 3.6), dpi=300)
    plt.hist2d(x, y, bins=100, cmap='viridis')
    # Add a colorbar
    plt.colorbar()

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Set axis labels
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_bar_aug():
    import matplotlib.pyplot as plt
    import numpy as np
    import scienceplots
    plt.style.use(['science', 'no-latex'])

    # Data
    categories = ['POMO', 'SGBS', 'AMDKD', 'Omni-VRP']
    no_aug_50 = [54.71, 35.36, 7.45, 21.60]
    aug_50 = [5.96, 2.72, 1.44, 4.45]
    # no_aug_100 = [91.00, 64.65, 10.11, 20.11]
    # aug_100 = [22.30, 22.78, 2.60, 9.07]

    # Set up the figure and axis
    fig, ax = plt.subplots(dpi=300, figsize=(3.5,4))

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    index = np.arange(len(categories))

    # Create the bar plots
    bar1 = ax.bar(index, no_aug_50, bar_width, label='No Augmentation')
    bar2 = ax.bar(index + bar_width, aug_50, bar_width, label='Augmentation')
    # bar3 = ax.bar(index + 2 * bar_width, no_aug_100, bar_width, label='No Augmentation (100)')
    # bar4 = ax.bar(index + 3 * bar_width, aug_100, bar_width, label='Augmentation (100)')

    # Set labels and title
    ax.set_xlabel('')
    ax.set_ylabel('Optimality gap (%)')
    ax.set_xticks(index + bar_width/2)
    ax.set_xticklabels(categories)
    ax.legend()

    # Show the plot
    plt.show()

def read_and_transform_to_tensor(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Remove newline characters and split each line into a list of floats
    data = [list(map(float, line.strip().split())) for line in content]

    # Split each list into x and y coordinates
    y_coordinates = [row[100:] for row in data]
    x_coordinates = [row[:100] for row in data]

    # Create a 3D tensor (line_num, 100, 2)
    tensor_data = torch.stack([torch.tensor(x_coordinates), torch.tensor(y_coordinates)], dim=2)

    return tensor_data


def plot_structure():
    POMO = np.array([0.1680, 0.3405, 0.3940, 0.4321, 0.4546])*100
    AMDKD = np.array([0.1267, 0.2602, 0.3196, 0.379, 0.4044])*100
    LKH = np.array([0.0839, 0.1093, 0.1479, 0.2217, 0.2859])*100

    import scienceplots
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(6.5,3.5), dpi=300)

    # X-axis values (assuming a simple range of 1 to 5 for this example)
    x_values = range(1, 6)

    # Plotting the data
    plt.plot(x_values, POMO, label='POMO')
    plt.plot(x_values, AMDKD, label='AMDKD')
    plt.plot(x_values, LKH, label='LKH')


    plt.scatter(x_values, POMO, s=3)
    plt.scatter(x_values, AMDKD, s=3)
    plt.scatter(x_values, LKH, s=3)

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Optimality gap (%)')

    plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])

    # Adding legend
    plt.legend()

    # Displaying the chart
    plt.show()


def plot_epoch():
    import scienceplots
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(6.5, 3.5), dpi=300)

    df = pd.read_csv('./TSP/result/sens/scores.csv')
    plt.plot(df['POMO']*100, label='POMO')
    plt.plot(df['SGBS']*100, label='SGBS')
    plt.plot(df['AMDKD']*100, label='AMDKD')
    plt.plot(df['OMNI-VRP']*100, label='OMNI-VRP')

    plt.legend(ncol=2)
    plt.xticks([0, 4, 9, 14, 19, 24, 29], [1, 5, 10, 15, 20, 25, 30])
    plt.xlabel('Epoch')
    plt.ylabel('Optimality gap (%)')
    plt.show()

def plot_epoch_evaluate():

    import scienceplots
    plt.style.use(['science', 'no-latex'])

    opt = np.array([7.7676, 4.2611, 3.2360, 5.0606, 5.4108])

    a5 = (np.array([7.7986, 4.3142, 3.3274, 5.1101, 5.4214]) - opt) / opt
    a10 = (np.array([7.8060, 4.3287, 3.3054, 5.1207, 5.4367]) - opt) / opt
    a15 = (np.array([7.7923, 4.3547, 3.4234, 5.1183, 5.4318]) - opt) / opt
    a20 = (np.array([8.1818, 4.4319, 3.3044, 5.2707, 5.5694]) - opt) / opt


    # Calculate average values
    avg_a5 = np.mean(a5)
    avg_a10 = np.mean(a10)
    avg_a15 = np.mean(a15)
    avg_a20 = np.mean(a20)


    data = pd.DataFrame({'Optimality gap (%)': np.concatenate([a5, a10, a15, a20]),
                         'Distribution': ['Uniform', 'GM', 'Diagonal', 'Cluster', 'Explosion']*4,
                         'T_p':[5]*5+[10]*5+[15]*5+[20]*5})
    import seaborn as sns
    plt.figure(dpi=300, figsize=(5, 4))
    ax = sns.barplot(data, x='T_p', y='Optimality gap (%)', hue='Distribution')

    tick_positions = ax.get_xticks().tolist()
    plt.plot(tick_positions, [avg_a5, avg_a10, avg_a15, avg_a20], label='Avg.Gap', color='black', linestyle='dashed')
    plt.legend()
    plt.xlabel(' ')

    plt.show()


def plot_attack_bar():
    # gnn_gaps = [0.1710, 0.1830, 1.2387]
    # matnet_gaps = [0.0134, 0.0215, 0.0297]
    # x_labels = ['Clean', 'Perturbation model', 'ASN']
    gnn_gaps = [17.10, 18.30, 123.87]
    x_labels = ['Clean', 'Perturbation', 'ASN']
    colors = ["#9BCB89", "#F7D58B", "#797BB7"]
    import scienceplots
    plt.style.use(['science', 'no-latex'])

    # Create bar chart
    plt.figure(figsize=(3, 4), dpi=300)
    bars = plt.bar(x_labels, gnn_gaps, color=colors)

    # Add labels and title
    plt.ylabel('Optimality Gap (%)')

    # Show plot
    plt.tight_layout()
    plt.show()


# def heuristic(problem):
#     parameters = {
#         'initial_location': 1
#     }
#
#     if isinstance(problem, torch.Tensor):
#         problem = problem.detach().cpu().numpy()
#     costs = []
#
#     for i in range(problem.shape[0]):
#         coords = problem[i, :, :]*1000
#         distances = compute_euclidean_distance_matrix(coords)
#         _, cost = farthest_insertion(distances, **parameters)
#         costs.append(cost/1000)
#     return costs


    # AT: [8078.0916957855225, 428.9816472530365, 546.7594158649445, 23426.06976699829, 23480.753847122192, 23071.456615447998, 23322.597326278687, 23685.67795562744, 111033.12994384766, 1460.9179215431213, 7919.053201675415, 677.194494009018]
    # POMO: [8343.023458480835, 429.48408102989197, 544.8838179111481, 23426.06976699829, 23333.186491012573, 23162.02018737793, 22928.026037216187, 23425.263786315918, 111155.7055053711, 1443.0244030952454, 7910.396196365356, 677.1096074581146]
    # gm3_10: [7663.881460189819, 430.88302302360535, 546.7594158649445, 23693.189975738525, 24015.37260246277, 22798.31958770752, 23184.601095199585, 23721.933067321777, 111852.91278076172, 1433.8993096351624, 7910.396196365356, 677.194494009018]

if __name__ == "__main__":
    # tensor_data = read_and_transform_to_tensor("F:\\program\\data_generate\\file.txt")
    # for i in [1, 2, 10, 20, 50]:
    #     plt.scatter(tensor_data[i, :, 0], tensor_data[i, :, 1])
    #     plt.show()
    # torch.save(tensor_data, "compression.pt")
    # def cal_rate():
    #     opt = 13.1373
    #     for i in [13.2720, 13.2987, 13.2719, 13.3874, 13.2838]:
    #         print((i - opt) / opt)
    # cal_rate()
    # dd = read_and_transform_to_tensor("F:\\program\\data_generate\\file.txt")
    # plot_bar_aug()

    # TSP
    # optimal = np.array([7542, 675, 1211, 7910, 21282, 22141, 20749, 21294, 22068, 59030, 96772, 26524, 26130, 42080, 2323, 3919, 80369, 49135, 2579, 48191])
    # pomo = np.array([7543, 675, 1234, 7910, 21367, 22216, 20785, 21472, 22165, 59390, 97804, 26707, 26435, 42512, 2502, 4094, 83380, 55352, 2865, 54529])
    # pomo_h = np.array([7547, 677, 1259, 7965, 21645, 22488, 21033, 21848, 22411, 59812, 99901, 27511, 26926, 43159, 2673, 4249, 84804, 58970, 2950, 55390])
    # pomo_g = np.array([7543, 676, 1270, 7924, 21458, 22308, 20876, 21548, 22256, 59030, 97359, 27054, 26514, 42652, 2526, 4063, 81736, 50791, 2843, 52567])
    # pomo_d = np.array([7585, 678, 1220, 8120, 21442, 22418, 20820, 21389, 22328, 59030, 97825, 27095, 26819, 42611, 2447, 4118, 83398, 56114, 2807, 52039])
    # at = np.array([7542, 675, 1214, 7910, 21343, 22293, 20749, 21672, 22165, 59088, 98127, 26657, 26328, 42501, 2422, 4063, 82528, 53284, 2808, 51991])
    #
    # print('HAM mean gap = ', np.round(100*(pomo_h-optimal)/optimal, 2))
    # print('POMO mean gap = ', np.round(100*(pomo - optimal) / optimal, 2))
    # print('diagonal mean gap = ', np.round(100*(pomo_d - optimal) / optimal, 2))
    # print('gm mean gap = ', np.round(100*(pomo_g - optimal) / optimal, 2))
    # print('at mean gap = ', np.round(100*(at - optimal) / optimal, 2))

    # CVRP
    # optimal = np.array(
    #     [27591, 12747, 55539, 15700, 43448, 45607, 47812, 25569, 44225, 58578, 19565, 117595, 40437, 82751, 38684, 26558, 75478,
    #      35291, 95151, 47161])
    # pomo = np.array(
    #         [29282, 13877, 58412, 16382, 47613, 50351, 52889, 26969, 50296, 62094, 20974, 122204, 43794, 89184, 41355, 28776, 83616,
    #          39952, 105343, 53937])
    # pomo_h = np.array(
    #         [29809, 13843, 58142, 16294, 47403, 51270, 52564, 26829, 50573, 62401, 21417, 120890, 43471, 89597, 41470, 28849, 84612,
    #          40171, 105165, 54163])
    # pomo_g = np.array(
    #         [29725, 13718, 58545, 16700, 46773, 50869, 52253, 26566, 49445, 61916, 21292, 121186, 43010, 88303, 41410, 29118, 82185,
    #          39034, 105238, 54074])
    # pomo_d = np.array(
    #         [30186, 15123, 58353, 16911, 47865, 50950, 53446, 27126, 50569, 61591, 21563, 121542, 44273, 88251, 41658, 29847, 82706,
    #          39820, 104618, 54985])
    # at = np.array(
    #         [29070, 13525, 58176, 16223, 46523, 49524, 52263, 26414, 49106, 61713, 21430, 121865, 43151, 87901, 41260, 29739, 82657,
    #          38604, 104149, 53530])
    # print('HAM mean gap = ', np.round(100*(pomo_h-optimal)/optimal, 2))
    # # print('POMO mean gap = ', np.round(100*(pomo - optimal) / optimal, 2))
    # print('gm mean gap = ', np.round(100*(pomo_g - optimal) / optimal, 2))
    # print('diagonal mean gap = ', np.round(100*(pomo_d - optimal) / optimal, 2))
    # # print('at mean gap = ', np.round(100*(at - optimal) / optimal, 2))
    plot_attack_bar()
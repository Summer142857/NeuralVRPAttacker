
import torch
import numpy as np
import glob

def get_random_problems(batch_size, problem_size, target_size, distribution="uniform"):
    if distribution == "uniform":
        return _get_random_problems(batch_size, problem_size, target_size)
    else:
        if target_size == 50:
            demand_scaler = 40
        elif target_size == 100:
            demand_scaler = 50
        else:
            raise NotImplementedError
        depot_xy = torch.rand(size=(batch_size, 1, 2))  # shape: (batch, 1, 2)
        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
        if distribution == "gaussian_mixture_2_5":
            problems = torch.tensor(generate_gaussian_mixture_tsp(batch_size, problem_size, num_modes=2, cdist=5), dtype=torch.float32)
            return depot_xy, problems, node_demand
        elif distribution == "gaussian_mixture_7_50":
            problems = torch.tensor(generate_gaussian_mixture_tsp(batch_size, problem_size, num_modes=7, cdist=50), dtype=torch.float32)
            return depot_xy, problems, node_demand
        elif distribution in ["uniform_rectangle", "gaussian", "cluster", "diagonal", "mixed"]:
            problems = torch.tensor(generate_tsp_dist(batch_size, problem_size, distribution), dtype=torch.float32)
            return depot_xy, problems, node_demand
        elif distribution == "cvrplib":
            return generate_tsp_dist(batch_size, problem_size, distribution)
        else:
            raise NotImplementedError


def _get_random_problems(batch_size, problem_size, target_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if target_size == 50:
        demand_scaler = 40
    elif target_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def generate_gaussian_mixture_tsp(dataset_size, graph_size, num_modes=0, cdist=0):
    '''
    Adaptation from AAAI-2022 "Learning to Solve Travelling Salesman Problem with Hardness-Adaptive Curriculum".
    '''

    def gaussian_mixture(graph_size=100, num_modes=0, cdist=1):
        '''
        GMM create one instance of TSP-100, using cdist
        '''
        from sklearn.preprocessing import MinMaxScaler
        nums = np.random.multinomial(graph_size, np.ones(num_modes) / num_modes)
        xy = []
        for num in nums:
            center = np.random.uniform(0, cdist, size=(1, 2))
            nxy = np.random.multivariate_normal(mean=center.squeeze(), cov=np.eye(2, 2), size=(num,))
            xy.extend(nxy)
        xy = np.array(xy)
        xy = MinMaxScaler().fit_transform(xy)
        return xy

    if num_modes == 0:  # (0, 0) - uniform
        return np.random.uniform(0, 1, [dataset_size, graph_size, 2])
    elif num_modes == 1 and cdist == 1:  # (1, 1) - gaussian
        return generate_tsp_dist(dataset_size, graph_size, "gaussian")
    else:
        res = []
        for i in range(dataset_size):
            res.append(gaussian_mixture(graph_size=graph_size, num_modes=num_modes, cdist=cdist))
        return np.array(res)

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

def generate_tsp_dist(n_samples, n_nodes, distribution):
    """
    Generate tsp instances with different distributions: ["cluster", "uniform_rectangle", "diagonal", "gaussian", "tsplib"]
    from "Generative Adversarial Training for Neural Combinatorial Optimization Models".
    """
    if distribution == "cluster":  # time-consuming
        x = []
        for i in range(n_samples):
            loc = []
            n_cluster = np.random.randint(low=2, high=9)
            loc.append(np.random.randint(1000, size=[1, n_cluster, 2]))
            prob = np.zeros((1000, 1000))
            coord = np.concatenate([np.tile(np.arange(1000).reshape(-1, 1, 1), [1, 1000, 1]),
                                    np.tile(np.arange(1000).reshape(1, -1, 1), [1000, 1, 1])], -1)
            for j in range(n_cluster):
                dist = np.sqrt(np.sum((coord - loc[-1][0, j, :]) ** 2, -1))
                dist = np.exp(-dist / 40)
                prob += dist
            for j in range(n_cluster):
                prob[loc[-1][0, j, 0], loc[-1][0, j, 1]] = 0
            prob = prob / prob.sum()
            index = np.random.choice(1000000, n_nodes - n_cluster, replace=False, p=prob.reshape(-1))
            coord = coord[index // 1000, index % 1000]
            loc.append(coord.reshape(1, -1, 2))
            loc = np.concatenate(loc, 1)
            x.append(loc)
        x = np.concatenate(x, 0) / 1000
    elif distribution == "uniform_rectangle":
        data = []
        for i in range(n_samples):
            width = np.random.uniform(0, 1)
            x1 = np.random.uniform(0, 1, [1, n_nodes, 1])
            x2 = np.random.uniform(0.5 - width / 2, 0.5 + width / 2, [1, n_nodes, 1])
            if np.random.randint(2) == 0:
                data.append(np.concatenate([x1, x2], 2))
            else:
                data.append(np.concatenate([x2, x1], 2))
        x = np.concatenate(data, 0)
    elif distribution == "diagonal":
        data = []
        for i in range(n_samples):
            x = np.random.uniform(low=0, high=1, size=(1, n_nodes, 1))
            r = np.random.uniform(low=0, high=1)
            if np.random.randint(4) == 0:
                x = np.concatenate([x, x * r + (1 - r) / 2], 2)
            elif np.random.randint(4) == 1:
                x = np.concatenate([x, (1 - x) * r + (1 - r) / 2], 2)
            elif np.random.randint(4) == 2:
                x = np.concatenate([x * r + (1 - r) / 2, x], 2)
            else:
                x = np.concatenate([(1 - x) * r + (1 - r) / 2, x], 2)
            width = np.random.uniform(low=0.05, high=0.2)
            x += np.random.uniform(low=-width / 2, high=width / 2, size=(1, n_nodes, 2))
            data.append(x)
        x = np.concatenate(data, 0)
    elif distribution == "gaussian":
        data = []
        for i in range(n_samples):
            mean = [0.5, 0.5]
            cov = np.random.uniform(0, 1)
            cov = [[1.0, cov], [cov, 1.0]]
            x = np.random.multivariate_normal(mean, cov, [1, n_nodes])
            data.append(x)
        x = np.concatenate(data, 0)
    elif distribution == "mixed":
        n_cluster_mix = 1
        center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(n_samples)])
        center = 0.2 + (0.8 - 0.2) * center
        std = 0.07
        for j in range(n_samples):
            mean_x, mean_y = center[j, ::2], center[j, 1::2]
            mutate_idx = np.random.choice(range(n_nodes), int(n_nodes / 2), replace=False)
            coords = torch.FloatTensor(n_nodes, 2).uniform_(0, 1)
            for i in range(n_cluster_mix):
                if i < n_cluster_mix - 1:
                    coords[mutate_idx[
                           int(n_nodes / n_cluster_mix / 2) * i:int(n_nodes / n_cluster_mix / 2) * (i + 1)]] = \
                        torch.cat((torch.FloatTensor(int(n_nodes / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                                   torch.FloatTensor(int(n_nodes / n_cluster_mix / 2), 1).normal_(mean_y[i], std)),
                                  dim=1)
                elif i == n_cluster_mix - 1:
                    coords[mutate_idx[int(n_nodes / n_cluster_mix / 2) * i:]] = \
                        torch.cat((torch.FloatTensor(int(n_nodes / 2) - int(n_nodes / n_cluster_mix / 2) * i,
                                                     1).normal_(mean_x[i], std),
                                   torch.FloatTensor(int(n_nodes / 2) - int(n_nodes / n_cluster_mix / 2) * i,
                                                     1).normal_(mean_y[i], std)), dim=1)
            coords = torch.where(coords > 1, torch.ones_like(coords), coords)
            coords = torch.where(coords < 0, torch.zeros_like(coords), coords).cuda()
            problems = coords.unsqueeze(0) if j == 0 else torch.cat((problems, coords.unsqueeze(0)), dim=0)
        return problems
    elif distribution in ["cvrplib"]:
        file_names = glob.glob("../CVRP/data/X/*.vrp")
        data = []
        demands = []
        depots = []
        name = []
        for file_name in file_names:
            with open(file_name, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if lines[i].strip().split(":")[0].split(" ")[0] == "DIMENSION":
                        nodes = int(lines[i].strip().split(" ")[-1])
                    if lines[i].strip().split(":")[0].split(" ")[0] == "CAPACITY":
                        capacity = int(lines[i].strip().split(" ")[-1])
                x = []
                d = []
                for i in range(len(lines)):
                    if lines[i].strip() == "NODE_COORD_SECTION":
                        for j in range(i + 1, i + nodes + 1):
                            line = [float(n) for n in lines[j].strip().split()]
                            assert j - i == int(line[0])
                            x.append([line[1], line[2]])
                        break
                for i in range(len(lines)):
                    if lines[i].strip() == "DEMAND_SECTION":
                        for j in range(i + 1, i + nodes + 1):
                            line = [float(n) for n in lines[j].strip().split()]
                            assert j - i == int(line[0])
                            d.append(float(line[1])/capacity)
                        break
                if len(x) == 0:
                    continue
                x = np.array(x)

                data.append(x[1:])
                depots.append(x[0])
                demands.append(d[1:])
                name.append(file_name.split('\\')[-1].split('.')[0])
        return depots, data, demands, name

    if distribution != "uniform_rectangle":
        x_min, x_max = x.min(1), x.max(1)
        x = x - x_min.reshape(-1, 1, 2)
        x = x / (x_max - x_min).max(-1).reshape(-1, 1, 1)
        x = x + (1 - x.max(1)).reshape(-1, 1, 2) / 2

    np.random.shuffle(x)

    assert x.shape[0] == n_samples
    assert x.shape[1] == n_nodes
    assert x.shape[2] == 2

    return x
import torch
from POMO.CVRPModel import CVRPModel as Model
from AMDKD.CVRPModel import CVRPModel as ADModel
from OMNI.CVRPModel import CVRPModel as OmniModel
from POMO.CVRPEnv import CVRPEnv as Env
from LKHSolver import get_lkh_solutions
from tqdm import tqdm
from utils import seed_torch
import time
import os
from POMO.CVRProblemDef import get_random_problems
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def model_test(model, env, batch_size, distribution, nodes_num, problems=None, return_gap=False, aug=8):
    model.eval()

    env.pomo_size, env.problem_size = nodes_num, nodes_num
    with torch.no_grad():
        env.load_problems(batch_size, aug, distribution=distribution, problems=problems)
        problems = (env.original_depot_xy, env.original_node_xy, env.original_node_demand)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

    # POMO Rollout
    ###############################################
    state, reward, done = env.pre_step()
    selected_list = torch.zeros(size=(batch_size*aug, env.pomo_size, 0))
    while not done:
        selected, _ = model(state)
        # shape: (batch, pomo)
        state, reward, done = env.step(selected, nodes_num)
        selected_list = torch.cat((selected_list, selected[:, :, None]), dim=2)

    # Return
    ###############################################
    aug_reward = reward.reshape(aug, batch_size, env.pomo_size)
    # shape: (augmentation, batch, pomo)

    max_pomo_reward, idx1 = aug_reward.max(dim=2)  # get best results from pomo
    # shape: (augmentation, batch)
    idx1 = idx1.unsqueeze(-1).expand(aug, batch_size, nodes_num)

    max_aug_pomo_reward, idx2 = max_pomo_reward.max(dim=0)  # get best results from augmentation

    if return_gap:
        optimal_score = torch.tensor(get_lkh_solutions(problems[0], problems[1], problems[2], nodes_num))
        gap = -max_aug_pomo_reward - optimal_score
        gap = torch.maximum(gap, torch.tensor(0)) / torch.tensor(optimal_score)
        return gap
    else:
        return -max_aug_pomo_reward


def benchmark(path, batch_size=1, node_num=100, model_name='POMO'):
    seed_torch()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if model_name == "AMDKD":
        model_params = {
            'type': 'AMDKD',
            'embedding_dim': 64,
            'sqrt_embedding_dim': 64 ** (1 / 2),
            'encoder_layer_num': 6,
            'qkv_dim': 8,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
            'critic_optimizer_params': {
                'lr': 1e-4
            },
            'actor_optimizer_params': {
                'lr': 5e-5
            },
        }
        model = ADModel(**model_params)
    else:
        model_params = {
            'embedding_dim': 128,
            'sqrt_embedding_dim': 128 ** (1 / 2),
            'encoder_layer_num': 6,
            'qkv_dim': 16,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
        }
        if model_name == "POMO":
            model = Model(**model_params)
        else:
            model = OmniModel(**model_params)
    env = Env(**{'problem_size': node_num, 'target_size': node_num})
    if 'AT' in path['path']:
        checkpoint_fullname = '{path}/checkpoint-{epoch}-critic.pt'.format(**path)
    else:
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**path)
    checkpoint = torch.load(checkpoint_fullname, map_location="cuda:0")
    model.load_state_dict(checkpoint['model_state_dict'])

    res = []
    times = []
    for distribution in ['cvrp']:
    # for distribution in ['uniform', 'gaussian_mixture_7_50', 'diagonal', 'cluster', 'explosion']:
    #     demands = torch.load('./dataset/demands/' + distribution + '_demand.pt').to('cuda:0')
    #     depots = torch.load('./dataset/depots/' + distribution + '_depot.pt').to('cuda:0')
    #     nodes = torch.load('./dataset/nodes/' + distribution + '_node.pt').to('cuda:0')
        import pickle

        def check_extension(filename):
            if os.path.splitext(filename)[1] != ".pkl":
                return filename + ".pkl"
            return filename

        with open(check_extension('./dataset/cvrp.pkl'), 'rb') as f:
            data = pickle.load(f)
            depots = torch.tensor([i[0] for i in data]).unsqueeze(1)
            nodes = torch.tensor([i[1] for i in data])
            demands = torch.tensor([i[2] for i in data]) / 50

        i = 0
        dis_res = torch.tensor([])
        start_time = time.time()
        for _ in tqdm(range(1000)):
            problem = (depots[i:i+batch_size], nodes[i:i+batch_size], demands[i:i+batch_size])
            dis_res = torch.concat([dis_res, torch.tensor(get_lkh_solutions(problem[0], problem[1], problem[2], node_num))])
            i += batch_size
            # dis_res = torch.concat([dis_res, model_test(model, env, batch_size, distribution, node_num, problem)])
        end_time = time.time()
        print('distribution = ' + distribution + ', time = ', end_time-start_time)
        res.append(dis_res)
        times.append(end_time-start_time)
    print(times)
    return res

def benchmark_vrplib(path, model_name='POMO'):
    seed_torch()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    depot_xy, problems, node_demand, names = get_random_problems(1, 100, 100, "cvrplib")
    if model_name == "AMDKD":
        model_params = {
            'type': 'AMDKD',
            'embedding_dim': 64,
            'sqrt_embedding_dim': 64 ** (1 / 2),
            'encoder_layer_num': 6,
            'qkv_dim': 8,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
            'critic_optimizer_params': {
                'lr': 1e-4
            },
            'actor_optimizer_params': {
                'lr': 5e-5
            },
        }
        model = ADModel(**model_params)
    else:
        model_params = {
            'embedding_dim': 128,
            'sqrt_embedding_dim': 128 ** (1 / 2),
            'encoder_layer_num': 6,
            'qkv_dim': 16,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
        }
        if model_name == "POMO":
            model = Model(**model_params)
        else:
            model = OmniModel(**model_params)
    if 'AT' in path['path']:
        checkpoint_fullname = '{path}/checkpoint-{epoch}-critic.pt'.format(**path)
    else:
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**path)
    checkpoint = torch.load(checkpoint_fullname, map_location="cuda:0")
    model.load_state_dict(checkpoint['model_state_dict'])

    res = []
    name = []
    for idx, instance in enumerate(problems):
        name.append(names[idx])
        instance = torch.Tensor(instance)
        loc_scaler = 1000
        instance = instance / loc_scaler
        instance = instance.unsqueeze(0)
        env = Env(**{'problem_size': instance.shape[1], 'target_size': instance.shape[1]})
        env.loc_scaler = loc_scaler
        rwd = model_test(model, env, 1, 'uniform',
                         instance.shape[1], (torch.Tensor(depot_xy[idx]).unsqueeze(0).unsqueeze(0)/loc_scaler,
                                             instance, torch.Tensor(node_demand[idx]).unsqueeze(0)))
        res.append(torch.round(loc_scaler*rwd).long().cpu().numpy())
    dis_res = [a[0] for a in res]
    res_dict = dict(zip(name, dis_res))
    return res_dict

def get_dataset(num=50):
    seed_torch(188)
    for distribution in ['gaussian_mixture_2_5']:
        env = Env(**{'problem_size': 100, 'target_size': 100})
        depots = torch.tensor([])
        nodes = torch.tensor([])
        demands = torch.tensor([])
        for i in range(20):
            env.load_problems(num, 1, distribution=distribution)
            depot, node, demand = env.original_depot_xy, env.original_node_xy, env.original_node_demand
            depots = torch.concat([depots, depot], dim=0)
            nodes = torch.concat([nodes, node], dim=0)
            demands = torch.concat([demands, demand], dim=0)
        torch.save(depots, distribution+'_depot'+'.pt')
        # torch.save(nodes, distribution + '_node' + '.pt')
        torch.save(demands, distribution + '_demand' + '.pt')

if __name__ == "__main__":
    res = benchmark({
            'path': './POMO/pretrained',
            'epoch': 30500,
        }, model_name="POMO")
    print([torch.mean(d) for d in res])
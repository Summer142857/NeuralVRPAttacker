from POMO.TSPModel import TSPModel as Model
from AMDKD.TSPModel import TSPModel as ADModel
from OMNI.TSPModel import TSPModel as OmniModel
from POMO.TSPEnv import TSPEnv as Env
from POMO.TSProblemDef import _get_random_problems
# from TSP.HacAttacker import HacAttacker as HAC
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
from utils import *

import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def model_test(model, env, batch_size, distribution, nodes_num, problems=None, return_gap=False, aug=8):
        model.eval()

        env.pomo_size, env.problem_size = nodes_num, nodes_num
        with torch.no_grad():
            env.load_problems(batch_size, aug, distribution=distribution, problems=problems)
            problems = env.original_problems
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state, None)

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
            optimal_score = torch.tensor(lkh(problems))
            gap = -max_aug_pomo_reward - optimal_score
            gap = torch.maximum(gap, torch.tensor(0)) / torch.tensor(optimal_score)
            return gap
        else:
            # return selected_list.gather(1, idx1)[idx2].squeeze(0)
            return -max_aug_pomo_reward.float()


def benchmark(path, batch_size=100, node_num=100, model_name='POMO'):
    torch.seed()
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
    env = Env(**{'problem_size': node_num})
    if 'AT' in path['path']:
        checkpoint_fullname = '{path}/checkpoint-{epoch}-critic.pt'.format(**path)
    else:
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**path)
    checkpoint = torch.load(checkpoint_fullname, map_location="cuda:0")
    print(path['epoch'])
    if 600 == path['epoch']:
        model.load_state_dict(checkpoint['solver_param']['model'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    res = []
    times = []
    # for distribution in ['gaussian_mixture_7_50']:
    for distribution in ['uniform', 'gaussian_mixture_7_50', 'diagonal', 'cluster', 'explosion']:
        data = torch.load('./dataset/'+distribution+'.pt').to('cuda:0')
        i = 0
        dis_res = torch.tensor([])
        start_time = time.time()
        for _ in tqdm(range(10)):
            problems = data[i:i+batch_size]
            i += batch_size
            dis_res = torch.concat([dis_res, model_test(model, env, batch_size, distribution, node_num, problems=problems, return_gap=False)])
            # dis_res = torch.concat([dis_res, torch.tensor(lkh(problems))], dim=0)
        end_time = time.time()
        print('distribution = ' + distribution + ', time = ', end_time-start_time)
        res.append(dis_res)
        times.append(end_time-start_time)
    print(times)
    return res


def benchmark_tsplib(path, batch_size=1, node_num=100, model_name='POMO'):
    torch.seed()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    samples, names = _get_random_problems(batch_size, node_num)
    if model_name == 'AMDKD':
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
        model = Model(**model_params)
    if 'AT' in path['path']:
        checkpoint_fullname = '{path}/checkpoint-{epoch}-critic.pt'.format(**path)
    else:
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**path)
    checkpoint = torch.load(checkpoint_fullname, map_location="cuda:0")
    if 600 == path['epoch']:
        model.load_state_dict(checkpoint['solver_param']['model'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    dis_res = []
    instance_name = []

    for idx, instance in enumerate(samples):
        # plt.scatter(instance[:, 0], instance[:, 1])
        # plt.show()
        instance = torch.tensor(instance, dtype=torch.float32)
        dist = compute_euclidean_distance_matrix(instance.squeeze(0).cpu().numpy())
        instance_name.append(names[idx])
        if names[idx] == "berlin52":
            pass
        # normalization
        loc_scaler = instance.max()
        instance = instance / loc_scaler
        instance = instance.unsqueeze(0)
        env = Env(**{'problem_size': instance.shape[1]})
        env.loc_scaler = loc_scaler

        # compute distance
        # rwd = model_test(model, env, batch_size, 'tsplib',  instance.shape[1], instance, return_gap=False).squeeze(0).type(torch.int64).cpu().numpy()
        rwd = model_test(model, env, batch_size, 'tsplib', instance.shape[1], instance, return_gap=False).squeeze(0).cpu().numpy()
        # cost = calculate_total_distance(tour, dist)
        cost = loc_scaler.cpu().numpy()*rwd

        dis_res.append(cost)
    res_dict = dict(zip(instance_name, dis_res))
    print(res_dict)
    return res_dict

# def benchmark_hac(batch_size=100, node_num=50):
#     torch.seed()
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     model_params = {
#         'embedding_dim': 128,
#         'sqrt_embedding_dim': 128 ** (1 / 2),
#         'encoder_layer_num': 6,
#         'qkv_dim': 16,
#         'head_num': 8,
#         'logit_clipping': 10,
#         'ff_hidden_dim': 512,
#         'eval_type': 'softmax',
#         'optimizer_params': {
#             'lr': 1e-4
#         },
#     }
#
#     running_params = {
#         'use_cuda': True,
#         'cuda_device_num': 0,
#         'model_load': {
#             'path': './TSP/POMO/pretrained',
#             'epoch': 3100,
#             },
#         'batch_size': batch_size,
#     }
#
#     env_params = {
#         'problem_size': node_num
#     }
#
#     model = Model(**model_params)
#     checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**running_params['model_load'])
#     checkpoint = torch.load(checkpoint_fullname, map_location="cuda:0")
#     model.load_state_dict(checkpoint['model_state_dict'])
#     env = Env(**env_params)
#
#     attacker = HAC(env_params, model_params, running_params)
#     res = torch.tensor([])
#     for _ in tqdm(range(10)):
#         data = attacker.get_hard_samples().to(attacker.device)
#         res = torch.concat([res, model_test(model, env, batch_size, 'uniform', node_num, problems=data)])
#     return res

def sampling():
    def bivariate_gaussian_pdf(x, y, mean, covariance):
        inv_covariance = np.linalg.inv(covariance)
        det_covariance = np.linalg.det(covariance)
        normalization = 1 / (2 * np.pi * np.sqrt(det_covariance))

        x_diff = x - mean[0]
        y_diff = y - mean[1]

        exponent = -0.5 * (x_diff * inv_covariance[0, 0] * x_diff + y_diff * inv_covariance[1, 1] * y_diff +
                           (x_diff * inv_covariance[0, 1] + y_diff * inv_covariance[1, 0]) * 2)

        return normalization * np.exp(exponent)


    # Number of points to sample
    num_samples = 100

    # Mean and covariance matrix for the Gaussian distribution
    mean = np.array([0.5, 0.5])
    covariance = np.array([[0.1, 0.03], [0.03, 0.2]])

    # Generate 100 uniformly distributed points in a 1x1 square
    uniform_points = np.random.rand(400, 2)

    # Calculate the PDF values for the uniform points from the Gaussian distribution
    pdf_values = bivariate_gaussian_pdf(uniform_points[:, 0], uniform_points[:, 1], mean, covariance)

    # Generate random values between 0 and the maximum PDF value
    threshold = np.max(pdf_values) * np.random.rand(400)

    # Accept points whose PDF values are greater than the threshold
    accepted_points = uniform_points[pdf_values > threshold][:num_samples]

    # Plot the results
    plt.scatter(uniform_points[:, 0], uniform_points[:, 1], label='Uniform Points')
    plt.scatter(accepted_points[:, 0], accepted_points[:, 1], color='red', label='Accepted Points from Gaussian', alpha=0.3)
    plt.legend()
    plt.show()
    plt.scatter(accepted_points[:, 0], accepted_points[:, 1], color='red', label='Accepted Points from Gaussian')
    plt.legend()
    plt.show()

def epoch_based_evaluation(batch_size=50, distribution="gaussian_mixture_3_10"):
    results_mean = torch.tensor([])
    results_std = torch.tensor([])

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
    for i in tqdm(range(0, 105, 5)):
        env = Env(**{'problem_size': 100})
        critic_ckpt_name = "checkpoint-" + str(int(i)) + "-critic.pt"
        critic_ckpt = "TSP/result/at/AT_100_400_epoch100/" + critic_ckpt_name
        model = Model(**model_params)
        checkpoint = torch.load(critic_ckpt, map_location="cuda:0")
        model.load_state_dict(checkpoint['model_state_dict'])

        epoch_result = torch.tensor([])
        for i in range(5):
            batch_result = model_test(model, env, batch_size, distribution, 100)
            epoch_result = torch.concat([epoch_result, batch_result])
        results_mean = torch.concat([results_mean, epoch_result.mean(0, keepdim=True)])
        results_std = torch.concat([results_std, epoch_result.std(0, keepdim=True)])
    return results_mean.cpu().numpy(), results_std.cpu().numpy()


def get_dataset(num=1000):
    for distribution in ['uniform', 'gaussian_mixture_2_5', 'gaussian_mixture_7_50', 'cluster', 'diagonal']:
        env = Env(**{'problem_size': 100})
        env.load_problems(num, 1, distribution=distribution)
        problems = env.original_problems
        torch.save(problems, distribution+'.pt')


if __name__ == "__main__":
    res = benchmark_tsplib({
            'path': './result/at/VHAC_n100',
            'epoch': 50,
        }, model_name='POMO')
    print(res)


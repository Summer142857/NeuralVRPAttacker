import torch
import os
from torch.utils.data import DataLoader
from POMO.CVRPModel import CVRPModel as Model
from AMDKD.CVRPModel import CVRPModel as ADModel
from SGBS.CVRPModel import CVRPModel as SGBSModel
from OMNI.CVRPModel import CVRPModel as OmniModel
from POMO.CVRPEnv import CVRPEnv as Env
import numpy as np
from torch.optim import Adam as Optimizer
import matplotlib.pyplot as plt
from utils import seed_torch, get_result_folder, LogData, create_logger
from tqdm import tqdm
from LKHSolver import get_lkh_solutions
from logging import getLogger
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def minmax(xy_):
    '''
    min max batch of graphs [b,n,2]
    '''
    xy_=(xy_-xy_.min(dim=1,keepdims=True)[0])/(xy_.max(dim=1,keepdims=True)[0]-xy_.min(dim=1,keepdims=True)[0])
    return xy_

def clip_grad_norms(param_groups, max_norm=10):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def get_final_output(data, env, model, batch_size, node_num, aug=False):
    if aug:
        env.load_problems(batch_size, problems=data, aug_factor=8)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected, node_num)

        aug_reward = reward.reshape(8, batch_size, env.pomo_size)
        max_aug_pomo_reward, _ = aug_reward.max(dim=0)
        return max_aug_pomo_reward.reshape(batch_size, env.pomo_size), 0
    else:
        env.load_problems(batch_size, problems=data)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        # probability for choosing a node
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)
        selected_list = torch.zeros(size=(batch_size, env.pomo_size, 0))

        state, reward, done = env.pre_step()
        while not done:
            selected, prob = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected, node_num)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            selected_list = torch.cat((selected_list, selected[:, :, None]), dim=2)
        log_prob = prob_list.log().sum(dim=2)
        return reward.reshape(batch_size, env.pomo_size), log_prob


class HacAttacker:
    def __init__(self,
                 pomo_params,
                 env_params,
                 model_params,
                 running_params):
        self.pomo_params = pomo_params
        self.env_params = env_params
        self.model_params = model_params
        self.running_params = running_params

        USE_CUDA = self.running_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.running_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        if self.model_params['type'] == "POMO":
            self.model = Model(**self.model_params)
        elif self.model_params['type'] == "AMDKD":
            self.model = ADModel(**self.model_params)
        elif self.model_params['type'] == "SGBS":
            self.model = SGBSModel(**self.model_params)
        elif self.model_params['type'] == "OMNI":
            self.model = OmniModel(**self.model_params)
        else:
            assert 0, 'Env_model not defined!'

        self.baseline = Model(**self.pomo_params)
        self.env = Env(**self.env_params)
        self.base_env = Env(**self.env_params)
        self.critic_env = Env(**self.env_params)

        self.batch_size = self.running_params['batch_size']
        self.problem_size = self.env_params['problem_size']

        model_load = running_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        if 600 == running_params['model_load']['epoch']:
            self.model.load_state_dict(checkpoint['solver_param']['model'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        model_load = running_params['pomo_model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        if 600 == running_params['pomo_model_load']['epoch']:
            self.baseline.load_state_dict(checkpoint['solver_param']['model'])
        else:
            self.baseline.load_state_dict(checkpoint['model_state_dict'])
        self.baseline_optimizer = Optimizer(self.baseline.parameters(),
                                          **self.pomo_params['optimizer_params'])
        self.baseline_optimizer.param_groups[0]['capturable'] = True

        self.base_optimizer = Optimizer(self.model.parameters(), **self.pomo_params['optimizer_params'])
        self.base_optimizer.param_groups[0]['capturable'] = True

        self.model_save_interval = 50

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()


    def get_hard_samples(self, eps=5, data=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if data is None:
            self.env.load_problems(batch_size)
            depot_xy, node_xy, node_demand = self.env.original_depot_xy, self.env.original_node_xy, self.env.original_node_demand
        else:
            depot_xy, node_xy, node_demand = data

        self.model.eval()
        self.baseline.eval()
        def get_hard(model,data,eps):
            self.update_baseline(data)
            data[1].requires_grad_()
            if self.baseline is not None:
                with torch.no_grad():
                    cost_b,_ = get_final_output(data, self.base_env, self.baseline, batch_size, self.problem_size)
                cost, ll = get_final_output(data, self.env, self.model, batch_size, self.problem_size)
                delta = torch.autograd.grad(eps*((cost/cost_b)*ll).mean(),data[1])[0]
            else:
                # As dividend is viewed as constant, it can be omitted in gradient calculation.
                cost, ll = get_final_output(data, self.env, self.model, batch_size, self.problem_size)
                delta = torch.autograd.grad(eps*(cost*ll).mean(), data[1])[0]
            ndata = data[1]+delta
            ndata = minmax(ndata)
            return ndata.detach().cpu()

        depotloader = DataLoader(depot_xy, batch_size=batch_size)
        nodeloader = DataLoader(node_xy, batch_size=batch_size)
        demandloader = DataLoader(node_demand, batch_size=batch_size)
        hard = torch.cat([get_hard(self.model, data, eps) for data in zip(depotloader, nodeloader, demandloader)],dim=0)
        return depot_xy, hard, node_demand

    def generate_hard(self, epoch):
        nodes = torch.tensor([], device=self.device)
        gap = torch.tensor([], device=self.device)
        for _ in tqdm(range(epoch)):
            new_hard = self.get_hard_samples()
            nodes = torch.concat([nodes, new_hard[1].to(self.device)])
            score, _ = get_final_output((new_hard[0].to(self.device),  new_hard[1].to(self.device), new_hard[2].to(self.device)), self.env, self.model, self.batch_size, self.problem_size, aug=True)
            score = score.max(dim=-1)[0]
            optimal_score = torch.tensor(get_lkh_solutions(new_hard[0].to(self.device),  new_hard[1].to(self.device), new_hard[2].to(self.device), self.env_params['problem_size']))
            optimal_gap = torch.maximum(-score - optimal_score, torch.tensor(0)) / torch.tensor(optimal_score)
            gap = torch.concat([gap, optimal_gap])
            print(gap.mean())
        return nodes, gap

    def update_baseline(self, problems):
        depots, nodes, demands = problems[0], problems[1], problems[2]
        batch, node_num, feature = nodes.shape[0], nodes.shape[1], nodes.shape[2]
        nodes = nodes.reshape((batch, node_num, feature))
        demands = demands.reshape((batch, node_num))
        train_num_episode = nodes.shape[0]
        episode = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.running_params['batch_size'], remaining)


            self.baseline.train()
            self.critic_env.load_problems(batch_size, problems=(depots[episode:episode+batch_size], nodes[episode:episode+batch_size], demands[episode:episode+batch_size]))
            reset_state, _, _ = self.critic_env.reset()
            self.baseline.pre_forward(reset_state)

            prob_list = torch.zeros(size=(batch_size, self.critic_env.pomo_size, 0))
            # shape: (batch, pomo, 0~problem)

            state, reward, done = self.critic_env.pre_step()
            while not done:
                selected, prob = self.baseline(state)
                # shape: (batch, pomo)
                state, reward, done = self.critic_env.step(selected, self.env_params['target_size'])
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            # Loss
            ###############################################
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            # shape: (batch, pomo)
            log_prob = prob_list.log().sum(dim=2)
            # size = (batch, pomo)
            loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
            # shape: (batch, pomo)
            loss_mean = loss.mean()

            # Score
            ###############################################
            max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo

            # Step & Return
            ###############################################
            self.baseline.zero_grad()
            loss_mean.backward()
            self.baseline_optimizer.step()
            episode += batch_size

    def train(self, num_epochs, bs=32):
        self.logger.info('=================================================================')
        for epoch in range(1, num_epochs+1):
            avg_score, avg_loss = self.train_epoch(bs, epoch)
            self.logger.info('Epoch {:3d}: Score: {:.4f},  Loss: {:.4f}'
                             .format(epoch, avg_score, avg_loss))
            if epoch % self.model_save_interval == 0 and epoch != 0:
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.base_optimizer.state_dict()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

    def train_epoch(self, batch_size1, epoch):
        episode = 0
        loop_cnt = 0
        num_episodes = self.running_params['num_episodes']
        scores, losses = [], []
        while episode < num_episodes:
            remaining = num_episodes - episode
            batch_size = min(batch_size1, remaining)
            avg_score, avg_loss = self.train_batch(batch_size, epoch)
            episode += batch_size
            scores.append(avg_score)
            losses.append(avg_loss)
            if epoch == 1:
                loop_cnt += 1
                if loop_cnt <= 20:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, num_episodes, 100. * episode / num_episodes,
                                             avg_score, avg_loss))
        return np.mean(scores), np.mean(losses)

    def train_batch(self, batch_size, epoch):
        node_xy = torch.rand(size=(batch_size, 100, 2))
        depot_xy = torch.rand(size=(batch_size, 1, 2))  # shape: (batch, 1, 2)
        node_demand = torch.randint(1, 10, size=(batch_size, 100)) / float(50)

        hard_data = self.get_hard_samples(data=(depot_xy, node_xy, node_demand), batch_size=batch_size)
        train_d = (torch.cat([depot_xy, hard_data[0].to(self.device)], dim=0),
                   torch.cat([node_xy, hard_data[1].to(self.device)], dim=0),
                   torch.cat([node_demand, hard_data[2].to(self.device)], dim=0))

        # Evaluate model, get costs and log probabilities
        cost, log_likelihood = get_final_output(train_d, self.env, self.model, train_d[0].size(0), self.problem_size)

        bl_val, _ = get_final_output(train_d, self.base_env, self.baseline, train_d[0].size(0), self.problem_size)

        loss = -((cost - bl_val) * log_likelihood).mean()
        # w = ((cost / bl_val) * log_likelihood).detach()
        # t = torch.FloatTensor([20 - (epoch % 20)]).to(loss.device)
        # w = torch.tanh(w)
        # w /= t
        # w = torch.nn.functional.softmax(w, dim=0)
        # loss = -((w * loss).mean()).sum()

        # Perform backward pass and optimization step
        self.base_optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(self.base_optimizer.param_groups, 10)
        self.base_optimizer.step()
        return -cost.mean().item(), loss.mean().item()

if __name__ == "__main__":
    pomo_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1 / 2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'optimizer_params': {
            'lr': 1e-4
        },
    }

    model_params = {
        'type': 'POMO',
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'softmax',
        'optimizer_params': {
            'lr': 1e-4
        }
    }

    running_params = {
        'use_cuda': True,
        'cuda_device_num': 0,
        'model_load': {
            'path': './POMO/pretrained',
            'epoch': 30500,
        },
        'pomo_model_load': {
            'path': './POMO/pretrained',
            'epoch': 30500,
        },
        'batch_size': 50,
        'num_episodes': 100*1000,
    }

    env_params = {
        'problem_size': 100,
        'target_size': 100
    }

    logger_params = {
        'log_file': {
            'desc': 'train_HAC'+'_tsp_n'+str(env_params['problem_size']),
            'filename': 'log.txt'
        }
    }
    create_logger(**logger_params)

    seed_torch()
    attacker = HacAttacker(pomo_params, env_params, model_params, running_params)
    attacker.train(100)
    # data, gap = attacker.generate_hard(20)
    # print("mean gap = ", gap.mean())
    # print("std gap = ", gap.std())
    # for i in [1, 6, 15, 38, 61]:
    #     plt.scatter(data.cpu()[i, :, 0], data.cpu()[i, :, 1])
    #     plt.show()

    # AT: 3.76  (10.11) (10.45)
    # HAC: 2.32 (10.88) (10.70)
    # GM: 4.85  (12.11) (12.62)
    # Diagonal: 3.61 () (20.47)
    # ASP: 4.07 (9.70)
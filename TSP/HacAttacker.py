import torch
import os
from torch.utils.data import DataLoader
from POMO.TSPModel import TSPModel as Model
from AMDKD.TSPModel import TSPModel as ADModel
from SGBS.TSPModel import TSPModel as SGBSModel
from OMNI.TSPModel import TSPModel as OmniModel
from POMO.TSPEnv import TSPEnv as Env
import numpy as np
from torch.optim import Adam as Optimizer
import matplotlib.pyplot as plt
from utils import lkh, seed_torch, get_result_folder, LogData, create_logger, hist2d_cities
from tqdm import tqdm
import math
from logging import getLogger

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
        model.pre_forward(reset_state, None)
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
        model.pre_forward(reset_state, None)

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
        if data is None:
            data = torch.FloatTensor(np.random.uniform(size=(self.batch_size, self.problem_size, 2)))
        if batch_size is None:
            batch_size = self.batch_size
        self.model.eval()
        self.baseline.eval()
        def get_hard(model,data,eps):
            data = data.to(self.device)
            self.update_baseline(data)
            data.requires_grad_()
            if self.baseline is not None:
                with torch.no_grad():
                    cost_b,_ = get_final_output(data, self.base_env, self.baseline, batch_size, self.problem_size)
                cost, ll = get_final_output(data, self.env, self.model, batch_size, self.problem_size)
                delta = torch.autograd.grad(eps*((cost/cost_b)*ll).mean(),data)[0]
            else:
                # As dividend is viewed as constant, it can be omitted in gradient calculation.
                cost, ll = get_final_output(data, self.env, self.model, batch_size, self.problem_size)
                delta = torch.autograd.grad(eps*(cost*ll).mean(), data)[0]
            ndata = data+delta
            ndata = minmax(ndata)
            return ndata.detach().cpu()
        dataloader = DataLoader(data, batch_size=batch_size)
        hard = torch.cat([get_hard(self.model, data, eps) for data in dataloader],dim=0)
        return hard

    def generate_hard(self, epoch):
        hard = torch.tensor([], device=self.device)
        gap = torch.tensor([], device=self.device)
        for _ in tqdm(range(epoch)):
            new_hard = self.get_hard_samples().to(self.device)
            hard = torch.concat([hard, new_hard])
            score, _ = get_final_output(new_hard, self.env, self.model, self.batch_size, self.problem_size, aug=True)
            score = score.max(dim=-1)[0]
            optimal_score = torch.tensor([lkh(new_hard)]).transpose(1, 0).squeeze(1)
            optimal_gap = torch.maximum(-score - optimal_score, torch.tensor(0)) / torch.tensor(optimal_score)
            gap = torch.concat([gap, optimal_gap])
        return hard, gap

    def update_baseline(self, problems):
        batch, node_num, feature = problems.shape[0], problems.shape[1], problems.shape[2]
        problems = problems.reshape((batch, node_num, feature))
        train_num_episode = problems.shape[0]
        episode = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.running_params['batch_size'], remaining)

            self.baseline.train()
            self.critic_env.load_problems(batch_size, problems=problems[episode:episode+batch_size])
            reset_state, _, _ = self.critic_env.reset()
            self.baseline.pre_forward(reset_state, ninf_mask=None)

            prob_list = torch.zeros(size=(batch_size, self.critic_env.pomo_size, 0))
            # shape: (batch, pomo, 0~problem)

            state, reward, done = self.critic_env.pre_step()
            while not done:
                selected, prob = self.baseline(state)
                # shape: (batch, pomo)
                state, reward, done = self.critic_env.step(selected, self.problem_size)
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
        data = torch.FloatTensor(np.random.uniform(size=(batch_size, self.problem_size, 2)))

        hard_data = self.get_hard_samples(data=data, batch_size=batch_size)
        train_d = torch.cat([data, hard_data], dim=0).to(self.device)

        # Evaluate model, get costs and log probabilities
        cost, log_likelihood = get_final_output(train_d, self.env, self.model, train_d.size(0), self.problem_size)

        bl_val, _ = get_final_output(train_d, self.base_env, self.baseline, train_d.size(0), self.problem_size)

        loss = ((cost - bl_val) * log_likelihood).mean()
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

def main_attack():
    pomo_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1 / 2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'softmax',
        'optimizer_params': {
            'lr': 1e-4
        },
    }

    # model_params = {
    #     'type': 'AMDKD',
    #     'embedding_dim': 64,
    #     'sqrt_embedding_dim': 64 ** (1 / 2),
    #     'encoder_layer_num': 6,
    #     'qkv_dim': 8,
    #     'head_num': 8,
    #     'logit_clipping': 10,
    #     'ff_hidden_dim': 512,
    #     'eval_type': 'softmax',
    #     'critic_optimizer_params': {
    #         'lr': 1e-4
    #     },
    #     'actor_optimizer_params': {
    #         'lr': 5e-5
    #     },
    # }

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
        'sgbs_beta': 10,  # beam_width of simulation guided beam search
        'sgbs_gamma_minus1': (10 - 1),  # for sbgs
        'optimizer_params': {
            'lr': 1e-4
        }
    }

    running_params = {
        'use_cuda': True,
        'cuda_device_num': 0,
        'model_load': {
            'path': './result/at/VHAC_n100',
            'epoch': 50,
        },
        'pomo_model_load': {
            'path': './POMO/pretrained',
            'epoch': 3100,
        },
        'batch_size': 50,
    }

    env_params = {
        'problem_size': 100
    }

    seed_torch()
    attacker = HacAttacker(pomo_params, env_params, model_params, running_params)
    data, gap = attacker.generate_hard(2)

    import scienceplots
    plt.style.use(['science', 'no-latex'])

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

    print("mean gap = ", gap.mean())
    print("std gap = ", gap.std())
    for i in [1, 6, 15, 38, 61]:
        plt.scatter(data.cpu()[i, :, 0], data.cpu()[i, :, 1])
        plt.show()

if __name__ == "__main__":
    # pomo_params = {
    #     'embedding_dim': 128,
    #     'sqrt_embedding_dim': 128 ** (1 / 2),
    #     'encoder_layer_num': 6,
    #     'qkv_dim': 16,
    #     'head_num': 8,
    #     'logit_clipping': 10,
    #     'ff_hidden_dim': 512,
    #     'eval_type': 'softmax',
    #     'optimizer_params': {
    #         'lr': 1e-4
    #     },
    # }
    #
    # model_params = {
    #     'type': 'POMO',
    #     'embedding_dim': 128,
    #     'sqrt_embedding_dim': 128 ** (1 / 2),
    #     'encoder_layer_num': 6,
    #     'qkv_dim': 16,
    #     'head_num': 8,
    #     'logit_clipping': 10,
    #     'ff_hidden_dim': 512,
    #     'eval_type': 'softmax',
    #     'optimizer_params': {
    #         'lr': 1e-4
    #     }
    # }
    #
    # running_params = {
    #     'use_cuda': True,
    #     'cuda_device_num': 0,
    #     'model_load': {
    #         'path': './result/VHAC_n100',
    #         'epoch': 50,
    #     },
    #     'pomo_model_load': {
    #         'path': './POMO/pretrained',
    #         'epoch': 3100,
    #     },
    #     'batch_size': 32,
    #     'num_episodes': 1000*100,
    # }
    #
    # env_params = {
    #     'problem_size': 100
    # }
    #
    # logger_params = {
    #     'log_file': {
    #         'desc': 'train_HAC'+'_tsp_n'+str(env_params['problem_size']),
    #         'filename': 'log.txt'
    #     }
    # }
    # create_logger(**logger_params)
    #
    # seed_torch()
    # attacker = HacAttacker(pomo_params, env_params, model_params, running_params)
    # attacker.train(100)
    main_attack()

    # GM: 11.54 (19.38) (19.10)
    # Diagonal: 23.01 () 36.48
    # AT: 5.19 (4.93) (3.80)
    # HAC: 3.25 (5.45) (12.83)
    # VHAC: 13.58 (3.19)

from logging import getLogger
import copy

import pandas as pd
import torch
from torch.optim import Adam as Optimizer
from utils import *
import os
from MatNet.ATSPModel import ATSPModel as Model
from pyCombinatorial.algorithm import farthest_insertion, nearest_neighbour, nearest_insertion
from MatNet.ATSPEnv import ATSPEnv as Env
from tqdm import tqdm
import random


os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MatNetAttacker:
    def __init__(self,
                 env_params,
                 base_params,
                 model_params,
                 running_params):
        self.env_params = env_params
        if base_params is None:
            self.base_params = model_params
        else:
            self.base_params = base_params
        self.model_params = model_params
        self.attacker_params = model_params
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
        self.plot_counter = 0

        # ENV and MODELS

        self.critic_env = Env(**{'problem_size': self.running_params['keep_node_num']})
        self.test_env = Env(**env_params)
        self.env = Env(**self.env_params)

        self.base_model = Model(**self.model_params)
        self.base_model.model_params['one_hot_seed_cnt'] = running_params['keep_node_num']
        model_load = running_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.base_model.load_state_dict(checkpoint['model_state_dict'])

        self.env.pomo_size = self.running_params['keep_node_num']

        self.actor_model = Model(**self.model_params)
        self.actor_model.model_params['one_hot_seed_cnt'] = env_params['problem_size']
        self.critic_model = Model(**self.model_params)
        self.critic_model.model_params['one_hot_seed_cnt'] = running_params['keep_node_num']
        # self.critic_model = ADModel(**critic_params)

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()


        if running_params['critic_load'] is not None:
            model_load = running_params['critic_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            print("loading critic model...")
            self.critic_model.load_state_dict(checkpoint['model_state_dict'])

        if running_params['actor_load'] is not None:
            model_load = running_params['actor_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}-actor.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            print("loading actor model...")
            self.actor_model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = Optimizer(self.actor_model.parameters(), **self.attacker_params['actor_optimizer_params'])
        self.critic_optimizer = Optimizer(self.critic_model.parameters(), **self.attacker_params['critic_optimizer_params'])
        self.optimizer.param_groups[0]['capturable'] = True
        self.critic_optimizer.param_groups[0]['capturable'] = True

        self.score_record = []

        self.time_estimator = TimeEstimator()

        self.start_epoch = 1

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.running_params['epochs'] + 1):
            self.logger.info('=================================================================')

            if (epoch-1) % self.running_params['round_epoch'] == 0 and epoch != 1:
                if self.base_params['type'] != "Heuristics":
                    self.base_model.load_state_dict(self.critic_model.state_dict())
                    # re-initialize actor
                    self.actor_model = Model(**self.model_params)
                    self.optimizer = Optimizer(self.actor_model.parameters(),
                                               **self.attacker_params['actor_optimizer_params'])
                    self.optimizer.param_groups[0]['capturable'] = True


            # Train
            gap, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, gap)
            self.result_log.append('train_loss', epoch, train_loss)
            self.score_record.append(gap)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.running_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.running_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.running_params['epochs'])
            model_save_interval = self.running_params['model_save_interval']


            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.actor_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}-actor.pt'.format(self.result_folder, epoch))
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.critic_model.state_dict(),
                    'optimizer_state_dict': self.critic_optimizer.state_dict()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}-critic.pt'.format(self.result_folder, epoch))


            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
                df = pd.DataFrame({'score': self.score_record})
                df.to_csv('{}/scores-{}.csv'.format(self.result_folder, self.base_params['type']))

    def _train_one_epoch(self, epoch):

        gap_AM = AverageMeter()
        loss_AM = AverageMeter()
        div_AM = AverageMeter()

        train_num_episode = self.running_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.running_params['batch_size'], remaining)

            avg_score, avg_loss, avg_diversity = self._train_one_batch(batch_size)
            # avg_score, avg_loss = self._random_batch(batch_size)
            gap_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            div_AM.update(avg_diversity, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 15:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Loss: {:.4f}, Div:{:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             gap_AM.avg, loss_AM.avg, div_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}, Div:{:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 gap_AM.avg, loss_AM.avg, div_AM.avg))

        return gap_AM.avg, loss_AM.avg

    # def _random_batch(self, batch_size):
    #     self.env.load_problems(batch_size)
    #     new_problems = resample_nodes(problems, self.running_params['keep_node_num'])
    #     new_base_score = self.base_model_batch_test(batch_size, new_problems, self.running_params['keep_node_num'])
    #     optimal_score = lkh(new_problems)
    #     gap = new_base_score - torch.tensor(optimal_score)
    #     gap = gap / torch.tensor(optimal_score)
    #
    #     return new_problems, gap

    def update_critic(self, problems):
        batch, pomo, node_num, feature = problems.shape[0], problems.shape[1], problems.shape[2], problems.shape[3]
        problems = problems.reshape((batch*pomo, node_num, feature))

        train_num_episode = problems.shape[0]
        episode = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.running_params['batch_size'], remaining)

            self.critic_model.train()
            self.critic_env.load_problems(batch_size, problems=problems[episode:episode+batch_size])
            reset_state, _, _ = self.critic_env.reset()
            self.critic_model.pre_forward(reset_state, None)

            prob_list = torch.zeros(size=(batch_size, self.critic_env.pomo_size, 0))
            # shape: (batch, pomo, 0~problem)

            state, reward, done = self.critic_env.pre_step()
            while not done:
                selected, prob = self.critic_model(state)
                # shape: (batch, pomo)
                state, reward, done = self.critic_env.step(selected, self.running_params['keep_node_num'])
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
            self.critic_model.zero_grad()
            loss_mean.backward()
            self.critic_optimizer.step()
            episode += batch_size

    def _train_one_batch(self, batch_size):

        # test on POMO
        self.env.load_problems(batch_size)
        problems = self.env.problems
        # original_base_score = self.base_model_batch_test(batch_size, problems, self.env_params['problem_size'])

        # Prep
        ###############################################
        self.actor_model.train()

        self.env.load_problems_manual(problems)
        reset_state, _, _ = self.env.reset()
        self.actor_model.pre_forward(reset_state,None)

        # probability for choosing a node
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)
        selected_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))

        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.actor_model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected, self.running_params['keep_node_num'])
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            selected_list = torch.cat((selected_list, selected[:, :, None]), dim=2)

        # Loss
        ###############################################
        idx = selected_list[:, :, :, None].type(torch.int64).expand(batch_size, self.env.pomo_size, self.env.pomo_size, self.env.problem_size)
        idx_1 = selected_list[:, :, None, :].type(torch.int64).expand(batch_size, self.env.pomo_size, self.env.pomo_size, self.env.pomo_size)
        new_problems = problems[:, None, :, :].expand(batch_size, self.env.pomo_size, self.env.problem_size, self.env.problem_size).gather(2, idx).gather(3, idx_1)
        new_base_scores = []
        for i in range(new_problems.shape[1]):
            new_base_scores.append(self.base_model_batch_test(batch_size, new_problems[:, i, :, :], self.running_params['keep_node_num']))
        new_problems_scores = torch.stack(new_base_scores, dim=1)

        self.update_critic(new_problems)
        critic_scores = []
        for i in range(new_problems.shape[1]):
            critic_scores.append(self.critic_model_batch_test(batch_size, new_problems[:, i, :, :], self.running_params['keep_node_num']))
        critic_scores = torch.stack(critic_scores, dim=1)
        # critic_scores = torch.tensor([lkh(new_problems[i]) for i in range(batch_size)])

        # optimal_score = torch.tensor([lkh(new_problems[i]) for i in range(batch_size)])
        gap = new_problems_scores - critic_scores
        gap = torch.maximum(gap, torch.tensor(0)) / torch.tensor(critic_scores)
        # top_score = torch.tensor([i[-1] for i in gap.sort(dim=1)[0]])

        # calculate optimal score
        if self.running_params['return_optimal_gap']:
            opt_idx = torch.tensor([i[-1] for i in gap.sort(dim=1)[1]])
            pp = new_problems.gather(1, opt_idx[:, None, None, None].expand(batch_size, 1, self.running_params['keep_node_num'], self.running_params['keep_node_num'])).squeeze(1).cpu().numpy()
            optimal_score = torch.tensor([lkh_atsp(pp)]).transpose(1,0)
            optimal_gap = new_problems_scores.gather(1, opt_idx.unsqueeze(1)) - optimal_score
            optimal_gap = torch.maximum(optimal_gap, torch.tensor(0)) / torch.tensor(optimal_score)
        else:
            optimal_gap = gap

        self.plot_counter += 1
        # if self.plot_counter % 10 == 0:
        #     draw_batch_instances(gap, new_problems)
        # diversity = get_diverse_metric(gap, new_problems, self.diverse_bin).mean()
        # gap = gap*torch.exp(self.running_params['lambda']*diversity)

        advantage = gap - gap.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Step & Return
        ###############################################
        self.actor_model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return optimal_gap.mean().item(), loss_mean.item(), torch.tensor(0)

    def base_model_batch_test(self, batch_size, problems, nodes_num):
        self.test_env.pomo_size, self.test_env.problem_size = problems.shape[1], problems.shape[1]
        self.base_model.eval()

        with torch.no_grad():
            self.test_env.load_problems_manual(problems)
            reset_state, _, _ = self.test_env.reset()
            self.base_model.pre_forward(reset_state, None)

        # POMO Rollout
        ###############################################

        state, reward, done = self.test_env.pre_step()
        while not done:
            selected, _ = self.base_model(state)
            # shape: (batch, pomo)
            state, reward, done = self.test_env.step(selected, nodes_num)

        # Return
        ###############################################
        aug_reward = reward.reshape(self.running_params['test_aug_factor'], batch_size, self.test_env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        return -max_aug_pomo_reward


    def critic_model_batch_test(self, batch_size, problems, nodes_num):
        self.critic_model.eval()

        self.critic_env.pomo_size, self.critic_env.problem_size = problems.shape[1], problems.shape[1]
        with torch.no_grad():
            self.critic_env.load_problems_manual(problems)
            reset_state, _, _ = self.critic_env.reset()
            self.critic_model.pre_forward(reset_state, None)

        # POMO Rollout
        ###############################################

        state, reward, done = self.critic_env.pre_step()
        while not done:
            selected, _ = self.critic_model(state)
            # shape: (batch, pomo)
            state, reward, done = self.critic_env.step(selected, nodes_num)

        # Return
        ###############################################
        aug_reward = reward.reshape(self.running_params['critic_aug_factor'], batch_size, self.critic_env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation

        return -max_aug_pomo_reward

    def generate(self, batch_size):
        self.actor_model.eval()
        self.critic_model.eval()

        self.env.load_problems(batch_size)
        problems = self.env.problems
        # original_base_score = self.base_model_batch_test(batch_size, problems, self.env_params['problem_size'])

        self.env.load_problems(batch_size, problems=problems)
        reset_state, _, _ = self.env.reset()
        self.actor_model.pre_forward(reset_state, None)

        # shape: (batch, pomo, 0~problem)
        selected_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))

        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.actor_model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected, self.running_params['keep_node_num'])
            selected_list = torch.cat((selected_list, selected[:, :, None]), dim=2)

        # Loss
        ###############################################
        idx = selected_list[:, :, :, None].type(torch.int64).expand(batch_size, self.env.pomo_size, self.env.pomo_size, self.env.problem_size)
        idx_1 = selected_list[:, :, None, :].type(torch.int64).expand(batch_size, self.env.pomo_size, self.env.pomo_size, self.env.pomo_size)
        new_problems = problems[:, None, :, :].expand(batch_size, self.env.pomo_size, self.env.problem_size, self.env.problem_size).gather(2, idx).gather(3, idx_1)
        new_base_scores = []
        for i in range(new_problems.shape[1]):
            new_base_scores.append(self.base_model_batch_test(batch_size, new_problems[:, i, :, :], self.running_params['keep_node_num']))
        new_problems_scores = torch.stack(new_base_scores, dim=1)

        critic_scores = []
        for i in range(new_problems.shape[1]):
            critic_scores.append(self.critic_model_batch_test(batch_size, new_problems[:, i, :, :], self.running_params['keep_node_num']))
        critic_scores = torch.stack(critic_scores, dim=1)

        gap = new_problems_scores - critic_scores
        gap = torch.maximum(gap, torch.tensor(0)) / torch.tensor(critic_scores)

        # calculate optimal score
        opt_idx = torch.tensor([i[-1] for i in gap.sort(dim=1)[1]])
        pp = new_problems.gather(1, opt_idx[:, None, None, None].expand(batch_size, 1,
                                                                        self.running_params['keep_node_num'],
                                                                        self.running_params['keep_node_num'])).squeeze(1).cpu().numpy()

        optimal_score = torch.tensor([lkh_atsp(pp)]).transpose(1,0)
        optimal_gap = new_problems_scores.gather(1, opt_idx.unsqueeze(1)) - optimal_score
        optimal_gap = optimal_gap / torch.tensor(optimal_score)

        self.logger.info('Batch Size: {:3d}: Test Score: {:.4f}'
                         .format(batch_size,optimal_gap.mean().cpu().numpy()))

        return pp, optimal_gap


def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":


    model_params = {
        'embedding_dim': 256,
        'sqrt_embedding_dim': 256 ** (1 / 2),
        'encoder_layer_num': 5,
        'qkv_dim': 16,
        'sqrt_qkv_dim': 16 ** (1 / 2),
        'head_num': 16,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'ms_hidden_dim': 16,
        'ms_layer1_init': (1/2)**(1/2),
        'ms_layer2_init': (1/16)**(1/2),
        'critic_optimizer_params': {
            'lr': 1e-4
        },
        'actor_optimizer_params': {
            'lr': 5e-5
        },
        'one_hot_seed_cnt': 20
    }

    # model_params = {
    #     'embedding_dim': 64,
    #     'sqrt_embedding_dim': 64 ** (1 / 2),
    #     'encoder_layer_num': 6,
    #     'qkv_dim': 8,
    #     'head_num': 8,
    #     'logit_clipping': 10,
    #     'ff_hidden_dim': 512,
    #     'eval_type': 'argmax',
    #     'critic_optimizer_params': {
    #         'lr': 1e-4
    #     },
    #     'actor_optimizer_params': {
    #         'lr': 5e-5
    # },
    # }

    # critic_params = {
    #     'type': 'AMDKD',
    #     'embedding_dim': 64,
    #     'sqrt_embedding_dim': 64 ** (1 / 2),
    #     'encoder_layer_num': 6,
    #     'qkv_dim': 8,
    #     'head_num': 8,
    #     'logit_clipping': 10,
    #     'ff_hidden_dim': 512,
    #     'eval_type': 'argmax',
    #     'critic_optimizer_params': {
    #         'lr': 1e-4
    #     },
    #     'actor_optimizer_params': {
    #         'lr': 5e-5
    #     },
    # }
    #
    # base_params = {
    #     'type': 'Heuristics',
    #     'name': 'nearest_insertion'
    # }

    base_params = {
        'type': 'POMO',
        'embedding_dim': 256,
        'sqrt_embedding_dim': 256 ** (1 / 2),
        'encoder_layer_num': 5,
        'qkv_dim': 16,
        'sqrt_qkv_dim': 16 ** (1 / 2),
        'head_num': 16,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'ms_hidden_dim': 16,
        'ms_layer1_init': (1/2)**(1/2),
        'ms_layer2_init': (1/16)**(1/2),
        'critic_optimizer_params': {
            'lr': 1e-4
        },
        'actor_optimizer_params': {
            'lr': 5e-5
        },
    }

    running_params = {
        'use_cuda': True,
        'cuda_device_num': 0,
        'model_load': {
            'path': './MatNet/result/saved_atsp50_model',
            'epoch': 8000,
        },
        'critic_load': {
            'path': './MatNet/result/saved_atsp50_model',
            'epoch': 8000,
        },
        'actor_load': None,
        # 'actor_load':{
        #     'path': './result/20231107_105421_train__tsp_n50_100epoch',
        #     'epoch': 1,
        # },
        'batch_size': 32,
        'augmentation_enable': True,
        'test_aug_factor': 1,
        'critic_aug_factor': 1,
        'aug_batch_size': 64,
        'train_episodes': 10*100,
        'epochs': 5,
        'model_save_interval': 5,
        'keep_node_num': 3,
        'return_optimal_gap': True,
        'round_epoch': 10,
        'AT': False
    }

    env_params = {
        'problem_size': 5
    }

    logger_params = {
        'log_file': {
            'desc': 'train_'+ base_params['type'] +'_atsp_n100_'+str(env_params['problem_size']),
            'filename': 'log.txt'
        }
    }
    create_logger(**logger_params)
    _print_config()

    # seed_torch()
    seed_torch(88)
    attacker = MatNetAttacker(env_params, base_params, model_params, running_params)
    attacker.run()
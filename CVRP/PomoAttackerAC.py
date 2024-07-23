from logging import getLogger
import copy

import torch
from torch.optim import Adam as Optimizer
from utils import *
import os
from .POMO.CVRPModel import CVRPModel as Model
from .AMDKD.CVRPModel import CVRPModel as ADModel
from .SGBS.CVRPModel import CVRPModel as SGBSModel
from .POMO.CVRPEnv import CVRPEnv as Env
from .OMNI.CVRPModel import CVRPModel as OmniModel
from .LKHSolver import get_lkh_solutions


os.environ['KMP_DUPLICATE_LIB_OK']='True'

class PomoAttacker:
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

        self.critic_env = Env(**{'problem_size': self.running_params['keep_node_num'], 'target_size': self.running_params['keep_node_num']})
        self.test_env = Env(**env_params)
        self.env = Env(**self.env_params)

        if self.base_params['type'] == "POMO":
            self.base_model = Model(**self.model_params)
        elif self.base_params['type'] == "AMDKD":
            self.base_model = ADModel(**self.base_params)
        elif self.base_params['type'] == "SGBS":
            self.base_model = SGBSModel(**self.base_params)
        elif self.base_params['type'] == "OMNI":
            self.base_model = OmniModel(**self.base_params)
        elif self.base_params['type'] == "Heuristics":
            pass
        else:
            assert 0, 'Env_model not defined!'
        self.env.pomo_size = self.running_params['keep_node_num']

        self.actor_model = Model(**self.model_params)
        self.critic_model = Model(**self.model_params)
        # self.critic_model = ADModel(**critic_params)

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # Restore
        if self.base_params['type'] != "Heuristics":
            model_load = running_params['model_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])

        if running_params['critic_load'] is not None:
            model_load = running_params['critic_load']
            try:
                checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
                checkpoint = torch.load(checkpoint_fullname, map_location=device)
            except:
                checkpoint_fullname = '{path}/checkpoint-{epoch}-critic.pt'.format(**model_load)
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
        self.distribution = "uniform"

        self.time_estimator = TimeEstimator()

        self.start_epoch = 1

        self.start_data_idx = 0

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
                    self.distribution = "uniform"

            # Train
            gap, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, gap)
            self.result_log.append('train_loss', epoch, train_loss)

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

    def _random_batch(self, batch_size):
        # demands = torch.load('./dataset/demands/' + 'uniform' + '_demand.pt').to('cuda:0')
        # depots = torch.load('./dataset/depots/' + 'uniform' + '_depot.pt').to('cuda:0')
        # nodes = torch.load('./dataset/nodes/' + 'uniform' + '_node.pt').to('cuda:0')
        # problem = (depots[self.start_data_idx:self.start_data_idx + batch_size], nodes[self.start_data_idx:self.start_data_idx + batch_size],
        #            demands[self.start_data_idx:self.start_data_idx + batch_size])
        self.start_data_idx += batch_size
        self.env.load_problems(batch_size)
        depot_xy, node_xy, node_demand = self.env.original_depot_xy, self.env.original_node_xy, self.env.original_node_demand

        # new_problems = resample_nodes((depot_xy, node_xy, node_demand), self.running_params['keep_node_num'], type="CVRP")
        new_problems = (depot_xy, node_xy, node_demand)

        new_base_score = self.base_model_batch_test(batch_size, new_problems, self.running_params['keep_node_num'])
        optimal_score = torch.tensor(get_lkh_solutions(new_problems[0], new_problems[1], new_problems[2], self.running_params['keep_node_num']))
        gap = new_base_score - torch.tensor(optimal_score)
        gap = gap / optimal_score
        self.logger.info('Batch Size: {:3d}: Test Score: {:.4f}'
                         .format(batch_size,gap.mean().cpu().numpy()))

        return new_problems[1], gap

    def update_critic(self, depots, nodes, demands):
        batch, pomo, node_num, feature = nodes.shape[0], nodes.shape[1], nodes.shape[2], nodes.shape[3]
        nodes = nodes.reshape((batch*pomo, node_num, feature))
        demands = demands.reshape((batch*pomo, node_num))
        train_num_episode = nodes.shape[0]
        episode = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.running_params['batch_size'], remaining)


            self.critic_model.train()
            self.critic_env.load_problems(batch_size, problems=(depots[episode:episode+batch_size], nodes[episode:episode+batch_size], demands[episode:episode+batch_size]))
            reset_state, _, _ = self.critic_env.reset()
            self.critic_model.pre_forward(reset_state)

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
        self.env.load_problems(batch_size, distribution=self.distribution)
        problems = (self.env.original_depot_xy, self.env.original_node_xy, self.env.original_node_demand)

        # Prep
        ###############################################
        self.actor_model.train()

        self.env.load_problems(batch_size, problems=problems)
        reset_state, _, _ = self.env.reset()
        self.actor_model.pre_forward(reset_state)

        # probability for choosing a node
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)
        selected_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))

        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.actor_model(state)
            # shape: (batch, pomo)
            state, _, done = self.env.step(selected, self.running_params['keep_node_num'], actor=True)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            selected_list = torch.cat((selected_list, selected[:, :, None]), dim=2)

        reshaped_selected_list = selected_list.view(-1, selected_list.size(2))
        # Use torch.unique along the second dimension to get unique values
        nodes = torch.unique(reshaped_selected_list, dim=1)
        nodes = torch.stack([i[torch.nonzero(i)] for i in nodes])
        selected_list = nodes.view(selected_list.size(0), selected_list.size(1), -1)

        idx = selected_list[:, :, :, None].expand(batch_size, self.env.pomo_size, self.running_params['keep_node_num'], 2).type(torch.int64) - 1

        new_nodes = problems[1][:, :, None, :].expand(batch_size, self.env.problem_size, self.env.pomo_size, 2).gather(1, idx)
        new_demands = problems[2][:, :, None].expand(batch_size, self.env.problem_size, self.env.pomo_size).gather(1, (selected_list-1).type(torch.int64))

        new_base_scores = []
        for i in range(new_nodes.shape[1]):
            new_base_scores.append(self.base_model_batch_test(batch_size, (problems[0], new_nodes[:, i, :, :], new_demands[:, i, :]), self.running_params['keep_node_num']))
        new_problems_scores = torch.stack(new_base_scores, dim=1)

        self.update_critic(problems[0].repeat(new_nodes.size(1),1,1), new_nodes, new_demands)
        critic_scores = []
        for i in range(new_nodes.shape[1]):
            critic_scores.append(self.critic_model_batch_test(batch_size, (problems[0], new_nodes[:, i, :, :], new_demands[:, i, :]), self.running_params['keep_node_num']))
        critic_scores = torch.stack(critic_scores, dim=1)

        gap = new_problems_scores - critic_scores
        gap = torch.maximum(gap, torch.tensor(0)) / torch.tensor(critic_scores)

        # if self.plot_counter % 10 == 0:
        #     draw_batch_instances(gap, new_problems)

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
        return gap.mean().item(), loss_mean.item(), torch.tensor(0)

    def base_model_batch_test(self, batch_size, problems, nodes_num):
        self.test_env.pomo_size, self.test_env.problem_size = nodes_num, nodes_num
        if self.base_params['type'] == "POMO" or self.base_params['type'] == "AMDKD" or self.base_params['type'] == "OMNI":
            self.base_model.eval()

            with torch.no_grad():
                self.test_env.load_problems(batch_size, self.running_params['test_aug_factor'], problems=problems)
                reset_state, _, _ = self.test_env.reset()
                self.base_model.pre_forward(reset_state)

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

            # for visualization
            # idx_pomo = aug_reward.argmax(dim=2).squeeze(0)[0:3]
            # for i, j in enumerate(idx_pomo):
            #     tour = self.test_env.selected_node_list[i, j]
            #     loc_with_depot = np.vstack((problems[0][i].cpu().numpy(), problems[1][i].cpu().numpy()))
            #     sorted_locs = loc_with_depot[np.concatenate(([0], tour.cpu().numpy(), [0]))]
            #     plt.scatter(sorted_locs[:, 0], sorted_locs[:, 1], c='red', marker='o', label='Cities')
            #     plt.plot(sorted_locs[:, 0], sorted_locs[:, 1], linestyle='-', linewidth=2, marker='o', markersize=8,
            #              color='blue', label='Tour')
            #     plt.show()

            return -max_aug_pomo_reward

        elif self.base_params['type'] == "SGBS":
            return self._test_one_batch_SGBS(batch_size, problems)
        elif self.base_params['type'] == "Heuristics":
            if isinstance(problems[0], torch.Tensor):
                problems = problems.detach().cpu().numpy()
            costs = []
            return torch.tensor(costs)

        else:
            assert 0, 'Env_model is not supported!'


    def _get_pomo_starting_points(self, model, env, num_starting_points):

        # Ready
        ###############################################
        model.eval()
        env.modify_pomo_size(self.env.pomo_size)
        env.reset()

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected, self.running_params['keep_node_num'])

        # starting points
        ###############################################
        sorted_index = reward.sort(dim=1, descending=True).indices
        selected_index = sorted_index[:, :num_starting_points]
        # shape: (batch, num_starting_points)
        selected_index = selected_index + 1

        return selected_index

    def _test_one_batch_SGBS(self, batch_size, problems):
        beam_width = self.base_params['sgbs_beta']
        expansion_size_minus1 = self.base_params['sgbs_gamma_minus1']
        rollout_width = beam_width * expansion_size_minus1
        aug_batch_size = self.running_params['test_aug_factor'] * batch_size

        # Ready
        ###############################################
        self.base_model.eval()
        self.test_env.load_problems(batch_size, self.running_params['test_aug_factor'], problems=problems)

        reset_state, _, __ = self.test_env.reset()
        self.base_model.pre_forward(reset_state)

        # POMO Starting Points
        ###############################################
        starting_points = self._get_pomo_starting_points(self.base_model, self.test_env, beam_width)

        # Beam Search
        ###############################################
        self.test_env.modify_pomo_size(beam_width)
        self.test_env.reset()

        # the first step, depot
        selected = torch.zeros(size=(aug_batch_size, self.test_env.pomo_size), dtype=torch.long)
        state, _, done = self.test_env.step(selected, beam_width)

        # the second step, pomo starting points
        state, _, done = self.test_env.step(starting_points, beam_width)

        # BS Step > 1
        ###############################################

        # Prepare Rollout-Env
        rollout_env = copy.deepcopy(self.test_env)
        rollout_env.modify_pomo_size(rollout_width)

        # LOOP
        first_rollout_flag = True
        while not done:

            # Next Nodes
            ###############################################
            probs = self.base_model.get_expand_prob(state)
            # shape: (aug*batch, beam, problem+1)
            ordered_prob, ordered_i = probs.sort(dim=2, descending=True)

            greedy_next_node = ordered_i[:, :, 0]
            # shape: (aug*batch, beam)

            if first_rollout_flag:
                prob_selected = ordered_prob[:, :, :expansion_size_minus1]
                idx_selected = ordered_i[:, :, :expansion_size_minus1]
                # shape: (aug*batch, beam, rollout_per_node)
            else:
                prob_selected = ordered_prob[:, :, 1:expansion_size_minus1+1]
                idx_selected = ordered_i[:, :, 1:expansion_size_minus1+1]
                # shape: (aug*batch, beam, rollout_per_node)

            # replace invalid index with redundancy
            next_nodes = greedy_next_node[:, :, None].repeat(1, 1, expansion_size_minus1)
            is_valid = (prob_selected > 0)
            next_nodes[is_valid] = idx_selected[is_valid]
            # shape: (aug*batch, beam, rollout_per_node)

            # Rollout to get rollout_reward
            ###############################################
            rollout_env.reset_by_repeating_bs_env(self.test_env, repeat=expansion_size_minus1)
            rollout_env_deepcopy = copy.deepcopy(rollout_env)  # Saved for later

            next_nodes = next_nodes.reshape(aug_batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)

            rollout_state, rollout_reward, rollout_done = rollout_env.step(next_nodes, self.running_params['keep_node_num'])
            while not rollout_done:
                selected, _ = self.base_model(rollout_state)
                # shape: (aug*batch, rollout_width)
                rollout_state, rollout_reward, rollout_done = rollout_env.step(selected, self.running_params['keep_node_num'])
            # rollout_reward.shape: (aug*batch, rollout_width)

            # mark redundant
            is_redundant = (~is_valid).reshape(aug_batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)
            rollout_reward[is_redundant] = float('-inf')

            # Merge Rollout-Env & BS-Env (Optional, slightly improves performance)
            ###############################################
            if first_rollout_flag is False:
                rollout_env_deepcopy.merge(self.test_env)
                rollout_reward = torch.cat((rollout_reward, beam_reward), dim=1)
                # rollout_reward.shape: (aug*batch, rollout_width + beam_width)
                next_nodes = torch.cat((next_nodes, greedy_next_node), dim=1)
                # next_nodes.shape: (aug*batch, rollout_width + beam_width)
            first_rollout_flag = False

            # BS Step
            ###############################################
            sorted_reward, sorted_index = rollout_reward.sort(dim=1, descending=True)
            beam_reward = sorted_reward[:, :beam_width]
            beam_index = sorted_index[:, :beam_width]
            # shape: (aug*batch, beam_width)

            self.test_env.reset_by_gathering_rollout_env(rollout_env_deepcopy, gathering_index=beam_index)
            selected = next_nodes.gather(dim=1, index=beam_index)
            # shape: (aug*batch, beam_width)
            state, reward, done = self.test_env.step(selected, self.running_params['keep_node_num'])


        # Return
        ###############################################
        aug_reward = reward.reshape(self.running_params['test_aug_factor'], batch_size, self.test_env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward = aug_reward.max(dim=2).values  # get best results from simulation guided beam search
        # shape: (augmentation, batch)

        max_aug_pomo_reward = max_pomo_reward.max(dim=0).values  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward  # negative sign to make positive value

        return aug_score

    def critic_model_batch_test(self, batch_size, problems, nodes_num):
        self.critic_model.eval()

        self.critic_env.pomo_size, self.critic_env.problem_size = nodes_num, nodes_num
        with torch.no_grad():
            self.critic_env.load_problems(batch_size, self.running_params['critic_aug_factor'], problems=problems)
            reset_state, _, _ = self.critic_env.reset()
            self.critic_model.pre_forward(reset_state)

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
        problems = (self.env.original_depot_xy, self.env.original_node_xy, self.env.original_node_demand)
        # original_base_score = self.base_model_batch_test(batch_size, problems, self.env_params['problem_size'])

        self.env.load_problems(batch_size, problems=problems)
        reset_state, _, _ = self.env.reset()
        self.actor_model.pre_forward(reset_state)

        # shape: (batch, pomo, 0~problem)
        selected_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))

        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.actor_model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected, self.running_params['keep_node_num'], actor=True)
            selected_list = torch.cat((selected_list, selected[:, :, None]), dim=2)

        reshaped_selected_list = selected_list.view(-1, selected_list.size(2))
        # Use torch.unique along the second dimension to get unique values
        nodes = torch.unique(reshaped_selected_list, dim=1)
        nodes = torch.stack([i[torch.nonzero(i)] for i in nodes])
        selected_list = nodes.view(selected_list.size(0), selected_list.size(1), -1)

        idx = selected_list[:, :, :, None].expand(batch_size, self.env.pomo_size, self.running_params['keep_node_num'], 2).type(torch.int64) - 1

        new_nodes = problems[1][:, :, None, :].expand(batch_size, self.env.problem_size, self.env.pomo_size, 2).gather(1, idx)
        new_demands = problems[2][:, :, None].expand(batch_size, self.env.problem_size, self.env.pomo_size).gather(1, (selected_list-1).type(torch.int64))

        new_base_scores = []
        for i in range(new_nodes.shape[1]):
            new_base_scores.append(self.base_model_batch_test(batch_size, (problems[0], new_nodes[:, i, :, :], new_demands[:, i, :]), self.running_params['keep_node_num']))
        new_problems_scores = torch.stack(new_base_scores, dim=1)

        critic_scores = []
        for i in range(new_nodes.shape[1]):
            critic_scores.append(self.critic_model_batch_test(batch_size, (problems[0], new_nodes[:, i, :, :], new_demands[:, i, :]), self.running_params['keep_node_num']))
        critic_scores = torch.stack(critic_scores, dim=1)

        gap = new_problems_scores - critic_scores
        gap = torch.maximum(gap, torch.tensor(0)) / torch.tensor(critic_scores)

        # storage
        # torch.save(problems[0], self.base_params['type']+'_'+self.env_params['env_params']+'depot.pt')
        # torch.save(new_nodes, self.base_params['type'] + '_' + self.env_params['env_params'] + 'node.pt')
        # torch.save(new_demands, self.base_params['type'] + '_' + self.env_params['env_params'] + 'demand.pt')
        # torch.save(new_problems_scores, self.base_params['type'] + '_' + self.env_params['env_params'] + 'score.pt')

        # calculate optimal score
        opt_idx = torch.tensor([i[-1] for i in gap.sort(dim=1)[1]])
        new_nodes = new_nodes.gather(1, opt_idx[:, None, None, None].expand(batch_size, 1, self.running_params['keep_node_num'], 2)).squeeze(1)
        new_demands = new_demands.gather(1, opt_idx[:, None, None].expand(batch_size, 1, self.running_params['keep_node_num'])).squeeze(1)
        optimal_score = torch.tensor(get_lkh_solutions(problems[0], new_nodes, new_demands, self.running_params['keep_node_num']))
        optimal_gap = new_problems_scores.gather(1, opt_idx.unsqueeze(1)).squeeze(1) - optimal_score
        optimal_gap = optimal_gap / optimal_score

        self.logger.info('Batch Size: {:3d}: Test Score: {:.4f}'
                         .format(batch_size,optimal_gap.mean().cpu().numpy()))

        return new_nodes, optimal_gap



def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":


    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1 / 2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
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

    # base_params = {
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

    base_params = {
        'type': 'POMO',
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'sgbs_beta': 4,  # beam_width of simulation guided beam search
        'sgbs_gamma_minus1': (4 - 1),  # for sbgs
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
            # 'path': './AMDKD/pretrained',
            # 'epoch': 100,
            'path': './POMO/pretrained',
            'epoch': 600,
            # 'path': './SGBS/pretrained',
            # 'epoch': 30500,
            # 'path': './OMNI/pretrained',
            # 'epoch': 250000,
        },
        'critic_load': {
            'path': './POMO/pretrained',
            'epoch': 30500,
        },
        'actor_load': None,
        # 'actor_load':{
        #     'path': './result/20231107_105421_train__tsp_n50_100epoch',
        #     'epoch': 1,
        # },
        'batch_size': 64,
        'augmentation_enable': True,
        'test_aug_factor': 8,
        'critic_aug_factor': 8,
        'aug_batch_size': 64,
        'train_episodes': 10*100,
        'epochs': 5,
        'model_save_interval': 5,
        'keep_node_num': 100,
        'return_optimal_gap': True,
        'round_epoch': 5
    }

    env_params = {
        'problem_size': 600,
        'target_size': running_params['keep_node_num']
    }

    logger_params = {
        'log_file': {
            'desc': 'train_'+ base_params['type'] +'_cvrp_asp_'+str(env_params['problem_size']),
            'filename': 'log.txt'
        }
    }
    create_logger(**logger_params)
    _print_config()

    seed_torch(88)
    attacker = PomoAttacker(env_params, base_params, model_params, running_params, base_params)
    attacker.run()
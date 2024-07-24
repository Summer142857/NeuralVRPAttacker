
from dataclasses import dataclass
import torch

from .TSProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, problem_size):

        # Const @INIT
        ####################################
        self.problem_size = problem_size
        self.pomo_size = problem_size

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)
        self.loc_scaler = None

    def load_problems(self, batch_size, aug_factor=1, distribution="uniform", problems=None):
        self.batch_size = batch_size

        if problems is None:
            self.problems = get_random_problems(batch_size, self.problem_size, distribution)
        else:
            self.problems = problems
        self.original_problems = self.problems
        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def modify_pomo_size(self, new_pomo_size):
        self.pomo_size = new_pomo_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, current_size):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        self.current_size = current_size
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.current_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        self.step_state.first_node = self.selected_node_list[:, :, 0]
        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.current_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.problem_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)
        if self.loc_scaler is not None:
            segment_lengths = torch.round(segment_lengths * self.loc_scaler) / self.loc_scaler

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def reset_by_repeating_bs_env(self, bs_env, repeat):
        self.selected_count = bs_env.selected_count
        self.current_node = bs_env.current_node.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.selected_node_list = bs_env.selected_node_list.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, 0~)

        # STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask = bs_env.step_state.ninf_mask.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, node)

    def reset_by_gathering_rollout_env(self, rollout_env, gathering_index):
        self.selected_count = rollout_env.selected_count
        self.current_node = rollout_env.current_node.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.selected_count)
        self.selected_node_list = rollout_env.selected_node_list.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, 0~)

        # STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.problem_size)
        self.step_state.ninf_mask = rollout_env.step_state.ninf_mask.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, problem)

        if gathering_index.size(1) != self.pomo_size:
            self.modify_pomo_size(gathering_index.size(1))

    def merge(self, other_env):
        self.current_node = torch.cat((self.current_node, other_env.current_node), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.selected_node_list = torch.cat((self.selected_node_list, other_env.selected_node_list), dim=1)
        # shape: (batch, pomo1 + pomo2, 0~)

        # STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo1 + pomo2)
        self.step_state.ninf_mask = torch.cat((self.step_state.ninf_mask, other_env.step_state.ninf_mask), dim=1)
        # shape: (batch, pomo1 + pomo2, problem)


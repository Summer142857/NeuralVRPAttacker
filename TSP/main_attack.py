import matplotlib.pyplot as plt

from PomoAttackerAC import PomoAttacker as Attacker
from tqdm import tqdm
from utils import *
from pyCombinatorial.algorithm import farthest_insertion, nearest_neighbour, nearest_insertion

def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    GENERATE_BATCH = 8
    Downsample_factor = 6

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

    # base_params = {
    #     'type': 'Heuristics',
    #     'name': 'farthest_insertion'
    # }

    base_params = {
        'type': 'GNN',
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'sgbs_beta': 5,  # beam_width of simulation guided beam search
        'sgbs_gamma_minus1': (5 - 1),  # for sbgs
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
        'batch_size': 64,
        'augmentation_enable': True,
        'test_aug_factor': 8,
        'critic_aug_factor': 8,
        'aug_batch_size': 64,
        'train_episodes': 10 * 100,
        'epochs': 100,
        'keep_node_num': 100,
    }

    running_params['critic_load'] = {
        'path': './result/20240313_101423_train_GNN_gnn_n100_600',
        'epoch': 5,
    }
    running_params['actor_load'] = {
        'path': './result/20240313_101423_train_GNN_gnn_n100_600',
        'epoch': 5,
    }

    if base_params['type'] == 'SGBS':
        running_params['model_load'] = {
            'path': './SGBS/pretrained',
            'epoch': 2800,
        }
    elif base_params['type'] == 'POMO':
        running_params['model_load'] = {
            'path': './POMO/pretrained',
            'epoch': 3100,
        }
    elif base_params['type'] == 'AMDKD':
        running_params['model_load'] = {
            'path': './AMDKD/pretrained',
            'epoch': 100,
        }
    elif base_params['type'] == 'OMNI':
        running_params['model_load'] = {
            'path': './OMNI/pretrained',
            'epoch': 250000,
        }
    elif base_params['type'] == 'GNN':
        running_params['model_load'] = {
            'path': './GNN/',
            'epoch': 0,
        }

    env_params = {
        'problem_size': int(running_params['keep_node_num']*Downsample_factor)
    }

    logger_params = {
        'log_file': {
            'desc': 'test_'+base_params['type']+'_tsp_n100_'+str(Downsample_factor),
            'filename': 'log.txt'
        }
    }
    create_logger(**logger_params)
    _print_config()

    seed_torch()
    attacker = Attacker(env_params, base_params, model_params, running_params)

    generated_problems = torch.tensor([])
    generated_gaps = torch.tensor([])
    for i in tqdm(range(20)):
        problems, gaps = attacker.generate(GENERATE_BATCH)
        # problems, gaps = attacker._random_batch(GENERATE_BATCH)
        generated_problems = torch.concat([generated_problems, problems])
        generated_gaps = torch.concat([generated_gaps, gaps.float()])
    print(torch.mean(generated_gaps), torch.std(generated_gaps))
    attacker.logger.info(
        'Test score mean: {:.4f}, Test score std: {:.4f}'.format(torch.mean(generated_gaps).cpu().numpy(),
                                                                 torch.std(generated_gaps).cpu().numpy()))
    # torch.save(generated_problems, base_params['type']+'.pt')
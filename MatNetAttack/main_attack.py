import matplotlib.pyplot as plt

from MatAttackerAC import MatNetAttacker as Attacker
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
        'batch_size': 64,
        'augmentation_enable': False,
        'test_aug_factor': 1,
        'critic_aug_factor': 1,
        'aug_batch_size': 64,
        'train_episodes': 10 * 100,
        'epochs': 100,
        'keep_node_num': 50,
    }


    running_params['critic_load'] = {
        'path': './result/20240326_001507_train_POMO_atsp_n50_200',
        'epoch': 5,
    }
    running_params['actor_load'] = {
        'path': './result/20240326_001507_train_POMO_atsp_n50_200',
        'epoch': 5,
    }

    running_params['model_load'] = {
        'path': './MatNet/result/saved_atsp50_model',
        'epoch': 8000,
    }


    env_params = {
        'problem_size': int(running_params['keep_node_num']*Downsample_factor)
    }

    logger_params = {
        'log_file': {
            'desc': 'test_'+base_params['type']+'_atsp_n50_'+str(Downsample_factor),
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
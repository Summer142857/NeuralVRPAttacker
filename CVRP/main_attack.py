
from PomoAttackerAC import PomoAttacker as Attacker
from tqdm import tqdm
from utils import *

def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

if __name__ == "__main__":
    GENERATE_BATCH = 10
    Downsample_factor = 4

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
        'critic_load': {
            'path': './result/actor_models/aug/CVRP100_'+str(Downsample_factor)+'/'+base_params['type'],
            'epoch': 5,
        },
        'actor_load': {
            'path': './result/actor_models/aug/CVRP100_'+str(Downsample_factor)+'/'+base_params['type'],
            'epoch': 5,
        },
        'batch_size': 64,
        'augmentation_enable': True,
        'test_aug_factor': 8,
        'critic_aug_factor': 8,
        'aug_batch_size': 64,
        'train_episodes': 10 * 100,
        'epochs': 100,
        'keep_node_num': 100,
    }
    if base_params['type'] == 'SGBS':
        running_params['model_load'] = {
            'path': './SGBS/pretrained',
            'epoch': 30500,
        }
    elif base_params['type'] == 'POMO':
        running_params['model_load'] = {
            'path': './POMO/pretrained',
            'epoch': 30500,
        }
    elif base_params['type'] == 'AMDKD':
        running_params['model_load'] = {
            'path': './AMDKD/pretrained',
            'epoch': 100,
        }
    elif base_params['type'] == "OMNI":
        running_params['model_load'] = {
            'path': './OMNI/pretrained',
            'epoch': 250000,
        }

    env_params = {
        # 'problem_size': int(running_params['keep_node_num'] * Downsample_factor),
        'problem_size': 100,
        'target_size': running_params['keep_node_num']
    }

    logger_params = {
        'log_file': {
            'desc': 'test_'+base_params['type']+'_cvrp_n100_'+str(Downsample_factor),
            'filename': 'log.txt'
        }
    }
    create_logger(**logger_params)
    _print_config()

    seed_torch(188)
    attacker = Attacker(env_params, base_params, model_params, running_params)

    generated_problems = torch.tensor([])
    generated_gaps = torch.tensor([])
    depots = torch.tensor([])
    nodes = torch.tensor([])
    demands = torch.tensor([])
    for i in tqdm(range(20)):
        # problems, gaps = attacker.generate(GENERATE_BATCH)
        problems, gaps = attacker._random_batch(GENERATE_BATCH)
        generated_problems = torch.concat([generated_problems, problems])
        generated_gaps = torch.concat([generated_gaps, gaps])
    #     depots = torch.concat([depots, depot_xy], dim=0)
    #     nodes = torch.concat([nodes, node_xy], dim=0)
    #     demands = torch.concat([demands, node_demand], dim=0)
    # distribution = 'uniform'
    # torch.save(depots, distribution + '_depot' + '.pt')
    # torch.save(nodes, distribution + '_node' + '.pt')
    # torch.save(demands, distribution + '_demand' + '.pt')
    print(torch.mean(generated_gaps), torch.std(generated_gaps))
    attacker.logger.info('Test score mean: {:.4f}, Test score std: {:.4f}'.format(torch.mean(generated_gaps).cpu().numpy(), torch.std(generated_gaps).cpu().numpy()))
    for i in [0, 3, 4, 5, 8]:
        plt.scatter(generated_problems.cpu()[i, :, 0], generated_problems.cpu()[i, :, 1])
        plt.show()
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
import argparse
from TSP.PomoAttackerAC import PomoAttacker as TSPAttacker
from CVRP.PomoAttackerAC import PomoAttacker as CVRPAttacker
from utils import *

def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

def run(args):

    assert args.problem in ["TSP", "CVRP"], "problem not support!"

    # parameters of env model
    if args.env in ["POMO", "OMNI", "SGBS"]:
        model_params = {
            'type': args.env,
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
        base_params = model_params
    elif args.env == "GNN":
        base_params = {
            'type': 'GNN',
            'embedding_dim': 128,
            'sqrt_embedding_dim': 128 ** (1 / 2),
            'encoder_layer_num': 6,
            'qkv_dim': 16,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
            'sgbs_beta': 10,  # beam_width of simulation guided beam search
            'sgbs_gamma_minus1': (10 - 1),  # for sbgs
            'critic_optimizer_params': {
                'lr': 1e-4
            },
            'actor_optimizer_params': {
                'lr': 5e-5
            },
        }
    elif args.env == "AMDKD":
        base_params = {
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
    elif args.env in ['nearest_insertion', 'farthest_insertion', 'nearest_insertion']:
        base_params = {
            'type': 'Heuristics',
            'name': args.env
        }
    else:
        raise "Env not implemented!"

    running_params = {
        'use_cuda': True,
        'cuda_device_num': 0,
        'actor_load': None,
        'batch_size': 64,
        'augmentation_enable': True,
        'test_aug_factor': 8,
        'critic_aug_factor': 8,
        'aug_batch_size': 64,
        'train_episodes': 10*100,
        'epochs': args.train_epoch,
        'model_save_interval': 5,
        'keep_node_num': args.target_size,
        'return_optimal_gap': True,
        'round_epoch': 10,
        'AT': False
    }

    if args.problem == "TSP":
        running_params['model_load'] = {
            'path': './TSP/POMO/pretrained',
            'epoch': 3100,
        }
    else:
        running_params['model_load'] = {
            'path': './CVRP/POMO/pretrained',
            'epoch': 30500,
        }

    if model_params['type'] == 'POMO':
        if args.problem == "TSP":
            running_params['critic_load'] = {
                'path': './TSP/POMO/pretrained',
                'epoch': 3100,
            }
        else:
            running_params['critic_load'] = {
                'path': './CVRP/POMO/pretrained',
                'epoch': 30500,
            }
    elif model_params['type'] == 'AMDKD':
        running_params['model_load'] = {
            'path': './CVRP/AMDKD/pretrained',
            'epoch': 100,
        }
    elif model_params['type'] == 'SGBS':
        running_params['model_load'] = {
            'path': './CVRP/SGBS/pretrained',
            'epoch': 30500,
        }
    elif model_params['type'] == "OMNI":
        running_params['model_load'] = {
            'path': './CVRP/OMNI/pretrained',
            'epoch': 250000,
        }

    env_params = {
        'problem_size': args.target_size*args.scale,
    }
    if args.problem == "CVRP":
        env_params['target_size'] = args.target_size

    logger_params = {
        'log_file': {
            'desc': 'train_'+ model_params['type'] +'_n'+str(args.target_size)+'_'+str(env_params['problem_size']),
            'filename': 'log.txt'
        }
    }
    create_logger(**logger_params)
    _print_config()

    seed_torch(args.seed)
    if args.train:
        if args.problem == "TSP":
            attacker = TSPAttacker(env_params, base_params, model_params, running_params)
        else:
            attacker = CVRPAttacker(env_params, base_params, model_params, running_params)
        attacker.run()
    else:
        print("generating hard instances")
        running_params['critic_load'] = {
            'path': args.path_attacker,
            'epoch': args.epoch_attacker,
        }
        running_params['actor_load'] = {
            'path': args.path_attacker,
            'epoch': args.epoch_attacker,
        }
        if args.problem == "TSP":
            attacker = TSPAttacker(env_params, base_params, model_params, running_params)
        else:
            attacker = CVRPAttacker(env_params, base_params, model_params, running_params)
        generated_problems = torch.tensor([])
        generated_gaps = torch.tensor([])
        for i in range(args.generate_epoch):
            problems, gaps = attacker.generate(args.generate_batch)
            generated_problems = torch.concat([generated_problems, problems])
            generated_gaps = torch.concat([generated_gaps, gaps.float()])
        print(torch.mean(generated_gaps), torch.std(generated_gaps))
        attacker.logger.info(
            'Test score mean: {:.4f}, Test score std: {:.4f}'.format(torch.mean(generated_gaps).cpu().numpy(),
                                                                     torch.std(generated_gaps).cpu().numpy()))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trainer for the attacker")
    parser.add_argument('-train', type=bool, default=False,
                        help="Train the attacker or generate hard instances using the attacker")
    parser.add_argument('--env', type=str, default="POMO", help="Name of the env model")
    parser.add_argument('--problem', type=str, default="TSP", choices=["TSP", "CVRP"])
    parser.add_argument("--target_size", type=int, default=50, help="Size of hard instances")
    parser.add_argument("--train_epoch", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--scale", type=int, default=2, help="Downsampling factor")
    parser.add_argument("--optimal_gap", type=bool, default=True,
                        help="Whether to show the optimal gap during training (need LKH)")
    parser.add_argument('-seed', type=int, default=88, help="Random seed")

    parser.add_argument('--path_attacker', type=str, default="./TSP/result/actor_models/aug/TSP50_2/POMO",
                        help="(Evaluation only) directory to load HMN and ASN")
    parser.add_argument('--epoch_attacker', type=int, default=5,
                        help="(Evaluation only) epoch index to load attacker")
    parser.add_argument('--generate_batch', type=int, default=20,
                        help="(Evaluation only) batch size for generating hard instances")
    parser.add_argument('--generate_epoch', type=int, default=20,
                        help="(Evaluation only) epochs for generating hard instances")

    run(args=parser.parse_args())

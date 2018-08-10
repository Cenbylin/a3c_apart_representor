from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import atari_env
from utils import read_config
from model import A3Clstm
from train import train
from train_rep import train_rep
from test import test
from shared_optim import SharedRMSprop, SharedAdam
from rnet import RepresentNet

#from gym.configuration import undo_logger_setup
import time

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=2,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='Pong-v0',
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--pre-rnet',
    default='None',
    metavar='PRN',
    help='the label of pre-trained rnet model.')
parser.add_argument(
    '--rep-train-time',
    type=int,
    default=200,
    metavar='rtt',
    help='representation learning times over traces.')
parser.add_argument(
    '--trace-length',
    type=int,
    default=200,
    metavar='tlen',
    help='size of traces sampled for training rnet.')
parser.add_argument(
    '--rl-r',
    type=float,
    default=1e-5,
    help='learning rate for training rnet.')
parser.add_argument(
    '--max-step',
    type=int,
    default=100000,
    help='what time to stop.')
parser.add_argument(
    '--actor-weight',
    type=float,
    default=0.5,
    help='what time to stop.')
parser.add_argument(
    '--log-target',
    help='corresponding to the log filename(ganme-{target}_log).')
# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior
# python main.py --env Pong-v0 --workers 7 --gpu-ids 0 --amsgrad True --pre-rnet 1wsam --rep-train-time 10 --trace-length 50 --log-target name
if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space, args.pre_rnet)
    if args.load:
        saved_state = torch.load(
            '{0}{1}_{2}.dat'.format(args.load_model_dir, args.env, args.log_target),
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
            optimizer_r = SharedAdam(
                shared_model.r_net.parameters(), lr=args.rl_r, amsgrad=args.amsgrad)
        optimizer.share_memory()
        optimizer_r.share_memory()
    else:
        optimizer = None

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    p = mp.Process(target=test, args=(args, shared_model, env_conf, lock, counter))
    p.start()
    processes.append(p)
    # p = mp.Process(target=train_rep, args=(args, shared_model, env_conf))
    # p.start()
    # processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, optimizer_r, env_conf, lock, counter))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    while counter.value<args.max_step:
        time.sleep(3)
    for p in processes:
        time.sleep(0.1)
        p.terminate()

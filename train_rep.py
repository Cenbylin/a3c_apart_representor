from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.nn as nn
from environment import atari_env
from model import A3Clstm
from player_util import Agent
from utils import ensure_shared_grads
import time
from torch.optim import Adam
import numpy as np

def train_rep(args, shared_model, env_conf):
    batch_size = 16
    train_times = args.rep_train_time
    trace = []
    td_class = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 9)]
    loss_fn = nn.CrossEntropyLoss()
    optimizer_r = Adam(shared_model.r_net.parameters(), lr=args.rl_r)
    optimizer_c = Adam(shared_model.c_net.parameters(), lr=args.rl_r)
    ptitle('Train rep')
    gpu_id = args.gpu_ids[-1]

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
            # player.model.r_net = player.model.r_net.cuda()
            # player.model.c_net = player.model.c_net.cuda()
    flag = True
    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.train()
            flag = False

        player.action_test()
        trace.append(player.state)
        if len(trace)>args.trace_length:
            # 训练几百次
            for _ in range(train_times):
                range_c = np.random.randint(0, len(td_class))
                TD = np.random.randint(td_class[range_c][0], td_class[range_c][1])
                begin = np.random.randint(0, len(trace) - TD - batch_size)
                former = torch.stack(trace[begin:begin + batch_size], dim=0)
                latter = torch.stack(trace[begin + TD:begin + TD + batch_size], dim=0)
                target = torch.zeros(batch_size, dtype=torch.long) + range_c
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        former = former.cuda()
                        latter = latter.cuda()
                        target = target.cuda()

                rep_f, rep_l = player.model.r_net(former), player.model.r_net(latter)
                output = player.model.c_net(rep_f, rep_l, False)
                loss = loss_fn(output, target)
                optimizer_r.zero_grad()
                optimizer_c.zero_grad()
                loss.backward()
                ensure_shared_grads(player.model.r_net, shared_model.r_net, gpu=gpu_id >= 0)
                ensure_shared_grads(player.model.c_net, shared_model.c_net, gpu=gpu_id >= 0)
                optimizer_r.step()
                optimizer_c.step()
            trace = []
        if player.done and not player.info:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            flag = True

            state = player.env.reset()
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

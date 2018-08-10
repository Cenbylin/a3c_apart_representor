from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
from rnet import RepresentNet


def train(rank, args, shared_model, optimizer, optimizer_r, env_conf, lock, counter):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = [Variable(torch.zeros(1, 512).cuda()), Variable(torch.zeros(1, 512).cuda())]
                    player.hx = [Variable(torch.zeros(1, 512).cuda()), Variable(torch.zeros(1, 512).cuda())]
            else:
                player.cx = [Variable(torch.zeros(1, 512)), Variable(torch.zeros(1, 512))]
                player.hx = [Variable(torch.zeros(1, 512)), Variable(torch.zeros(1, 512))]
        else:
            player.cx = [Variable(player.cx[0].data), Variable(player.cx[1].data)]
            player.hx = [Variable(player.hx[0].data), Variable(player.cx[1].data)]

        # 测试rnet的更新有没有影响到这里
        # ps = list(player.model.r_net.named_parameters())
        # n, v = ps[6]
        # print(v.sum()) 
        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx[0], player.cx[0]), (player.hx[1], player.cx[1])))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        with lock:
            counter.value += 1
        # rnet
        player.model.r_net.zero_grad()
        (args.actor_weight * policy_loss + (1-args.actor_weight) * value_loss).backward(retain_graph=True)
        ensure_shared_grads(player.model.r_net, shared_model.r_net, gpu=gpu_id >= 0)
        optimizer_r.step()
        
        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        player.model.r_net.zero_grad()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()

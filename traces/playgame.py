import pygame
import gym
import numpy as np
import argparse
import matplotlib.pylab as plt
from environment import atari_env
from utils import read_config

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--env',
    default='Pong-v0',  # SpaceInvaders-v0
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
args = parser.parse_args([])
setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]


# 方向键对应的值
# Left_s = 113
# Right_s = 101
# Left = 97
# Right = 100
# J = 106
# NO = 0
# keyboard_option = [Left_s, Right_s, Left, Right, J]
# key_action = {NO:0, Left_s:5, Right_s:4, Left:3, Right:2, J:1}
up_1 = 113
up_2 = 101
down_1 = 97
down_2 = 100
J = 106
NO = 0
keyboard_option = [up_1, up_2, down_1, down_2]
key_action = {NO:0, up_1:2, up_2:4, down_1:3, down_2:5}
# up 4. 2
# down 5 3
# no 0 1
# left 3 5.
def grayshow(img):
    img = img.squeeze()
    # img = img / 2 + 0.5  # unnormalize
    # npimg = img.numpy()
    plt.imshow(img, cmap='gray')
    plt.show()
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption('键盘监听中')
    screen.fill((255, 255, 255))
    pygame.key.set_repeat(70)
    pygame.display.flip()

    env = atari_env(args.env, env_conf, args)  # gym.make("SpaceInvaders-v0")

    # action1 = env.action_space.sample()
    # print(env.action_space.n)
    # while action1 in [0,2,3,4,5]:
    #     action1 = env.action_space.sample()
    # print(action1)
    init_log = 3
    while True:
        trace_s = []
        trace_a = []
        s = env.reset()
        action = key_action[NO]
        while True:
            # grayshow(s)
            trace_s.append(s)
            env.render()

            trace_a.append(action)
            s, r, done, _ = env.step(action)
            action = key_action[NO]
            skip = 3000
            # listening
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key in key_action.keys():
                        action = key_action[event.key]
                if skip > 0:
                    skip -= 1
                else:
                    break
            if done:
                # 保存
                t_s = np.stack(trace_s, axis=0)
                t_a = np.stack(trace_a, axis=0)
                print(t_s.shape)
                np.save("save/pong-s-{}".format(init_log), t_s)
                np.save("save/pong-a-{}".format(init_log), t_a)
                break

        init_log += 1
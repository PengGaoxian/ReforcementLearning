"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import gym
import tensorflow as tf
from RL_brain import Actor, Critic

# 超参数
OUTPUT_GRAPH = False # 是否输出tensorboard文件
MAX_EPISODE = 3000 # 运行的最大回合次数
DISPLAY_REWARD_THRESHOLD = 300  # 当回合总reward大于200时显示模拟窗口
MAX_EP_STEPS = 10000   # 一回合最大的步数
RENDER = False  # 默认不显示模拟窗口
GAMMA = 0.9     # 奖励衰减率
LR_A = 0.001    # Actor的学习率
LR_C = 0.01     # Critic的学习率

env = gym.make('CartPole-v0') # 创建CartPole-v0环境
env.seed(1)  # 设置环境的随机种子
env = env.unwrapped # 取消限制

N_F = env.observation_space.shape[0] # 状态空间的维度
N_A = env.action_space.n # 动作空间的维度


sess = tf.Session() # 创建tensorflow会话

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A) # 实例化Actor
critic = Critic(sess, n_features=N_F, lr=LR_C, gamma=GAMMA) # 实例化Critic

sess.run(tf.global_variables_initializer()) # 初始化tensorflow网络参数

# 输出tensorboard文件
if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

# 运行MAX_EPISODE个回合，记录回合总reward
for i_episode in range(MAX_EPISODE):
    s = env.reset() # 初始化环境
    t = 0 # 初始化步数
    track_r = [] # 记录回合中每一步的reward
    while True:
        # 如果回合总reward大于DISPLAY_REWARD_THRESHOLD，显示模拟窗口
        if RENDER:
            env.render()

        a = actor.choose_action(s) # 根据状态选择动作
        s_, r, done, info = env.step(a) # 执行动作，得到下一个状态、奖励、是否结束、信息

        # 将回合结束的最后一个动作的奖励增大
        if done:
            r = -20

        track_r.append(r) # 记录回合中每一步的reward

        td_error = critic.learn(s, r, s_)  # 学习更新Critic的网络参数，gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # 学习更新Actor的网络参数，true_gradient = grad[logPi(s,a) * td_error]

        s = s_ # 更新状态
        t += 1 # 更新步数

        # 如果回合结束或者步数达到最大值，输出回合总reward
        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r) # 计算回合总reward

            # 如果没有running_reward，将回合总reward赋值给running_reward
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05 # 计算所有回合平均的总reward

            # 如果平均总reward大于DISPLAY_REWARD_THRESHOLD，显示模拟窗口
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True  # rendering

            print("episode:", i_episode, "  reward:", int(running_reward))
            break

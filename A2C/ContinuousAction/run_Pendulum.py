"""
Actor-Critic with continuous action using TD-error as the Advantage, Reinforcement Learning.

The Pendulum example (based on https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb)

Cannot converge!!! oscillate!!!

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow r1.3
gym 0.8.0
"""

import tensorflow as tf
from RL_brain import Actor, Critic
import gym

OUTPUT_GRAPH = False # 是否输出tensorboard文件
MAX_EPISODE = 1000 # 最大回合数
MAX_EP_STEPS = 200 # 回合的最大步数
DISPLAY_REWARD_THRESHOLD = -100  # 显示模拟窗口的奖励阈值
RENDER = False  # 默认不显示模拟窗口
GAMMA = 0.9 # 奖励衰减率
LR_A = 0.001    # Actor的学习率
LR_C = 0.01     # Critic的学习率

env = gym.make('Pendulum-v0') # 创建Pendulum-v0环境
env.seed(1)  # 设置环境的随机种子
env = env.unwrapped # 取消限制

N_S = env.observation_space.shape[0] # 状态空间的维度
A_BOUND = env.action_space.high # 动作的上限

sess = tf.Session() # 创建tensorflow会话

actor = Actor(sess, n_features=N_S, lr=LR_A, action_bound=[-A_BOUND, A_BOUND]) # 创建Actor
critic = Critic(sess, n_features=N_S, lr=LR_C, gamma=GAMMA) # 创建Critic

sess.run(tf.global_variables_initializer()) # 初始化tensorflow的所有变量

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph) # 输出tensorboard文件

for i_episode in range(MAX_EPISODE):
    s = env.reset() # 初始化环境
    t = 0 # 记录每回合的步数
    ep_rs = [] # 记录每回合的奖励
    while True:
        if RENDER:
            env.render() # 显示模拟窗口

        a = actor.choose_action(s) # 选择动作
        s_, r, done, info = env.step(a) # 执行动作
        r /= 10 # 奖励的缩放

        # 学习更新Critic网络并计算td_error，即A(s)，gradient = grad[r + gamma * V(s_) - V(s)]
        td_error = critic.learn(s, r, s_)
        # 学习更新Actor网络，true_gradient = grad[logPi(s,a) * td_error]
        actor.learn(s, a, td_error)

        s = s_ # 更新状态
        t += 1 # 步数加1
        ep_rs.append(r) # 记录奖励

        if t > MAX_EP_STEPS:
            ep_rs_sum = sum(ep_rs) # 计算总奖励
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum # 第一次运行，设置running_reward为总奖励
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1 # 计算running_reward，即奖励的平均值

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True  # rendering

            print("episode:", i_episode, "  reward:", int(running_reward))
            break

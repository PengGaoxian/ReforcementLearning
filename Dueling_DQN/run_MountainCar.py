"""
控制Agent与环境交互

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

env = gym.make('MountainCar-v0') # 创建MountainCar-v0环境
env = env.unwrapped # 取消限制
env.seed(21) # 设置随机种子
MEMORY_SIZE = 10000 # 记忆库大小

sess = tf.Session() # 创建会话

with tf.variable_scope('natural_DQN'):
    # 创建DQN算法的实例
    RL_natural = DQNPrioritizedReplay(
        n_actions=3, # 动作数量
        n_features=2, # 特征（状态）所占数组元素的数量
        memory_size=MEMORY_SIZE, # 记忆库大小
        e_greedy_increment=0.00005, # e_greedy的增量
        sess=sess, # tensorflow的会话
        prioritized=False, # 是否使用Prior_Replay_DQN算法
    )

with tf.variable_scope('DQN_with_prioritized_replay'):
    # 创建Prior_Replay_DQN算法的实例
    RL_prio = DQNPrioritizedReplay(
        n_actions=3, # 动作数量
        n_features=2, # 特征（状态）所占数组元素的数量
        memory_size=MEMORY_SIZE, # 记忆库大小
        e_greedy_increment=0.00005, # e_greedy的增量
        sess=sess, # tensorflow的会话
        prioritized=True, # 是否使用Prior_Replay_DQN算法
        output_graph=True, # 是否输出tensorboard文件
    )

sess.run(tf.global_variables_initializer()) # 初始化tensorflow的变量


def run_agent(RL):
    total_steps = 0 # 记录所有回合的总步数
    steps = [] # 记录每回合结束的总步数（包括前面几个回合的步数）
    episodes = [] # 记录回合数

    # 进行20回合
    for i_episode in range(20):
        observation = env.reset() # 初始化环境
        while True:
            # env.render() # 显示动画

            action = RL.choose_action(observation) # 根据当前状态选择动作
            observation_, reward, done, info = env.step(action) # 执行动作

            if done:
                reward = 10

            RL.store_transition(observation, action, reward, observation_) # 存储记忆

            if total_steps > MEMORY_SIZE: # 如果记忆库已满，则开始学习
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_ # 更新状态
            total_steps += 1 # 总步数加1

    RL.show_parameters() # 显示DQN算法的参数

    return np.vstack((episodes, steps)) # 返回回合输和回合结束的总步数

his_natural = run_agent(RL_natural) # 使用DQN算法的每回合的步数
his_prio = run_agent(RL_prio) # 使用Prior_Replay_DQN算法的每回合的步数

# 画图显示每回合步数的变化
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total steps')
plt.xlabel('episode')
plt.grid()
plt.show()



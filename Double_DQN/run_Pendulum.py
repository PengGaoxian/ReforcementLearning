"""
控制Agent与环境交互
"""


import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make('Pendulum-v0') # 创建Pendulum-v0环境
env = env.unwrapped # 取消限制
env.seed(1) # 设置环境的随机种子

MEMORY_SIZE = 3000 # 定义记忆库的大小
ACTION_SPACE = 11 # 定义动作的数量
sess = tf.Session() # 创建tensorflow的session

# 实例化DQN算法类
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, # 动作的数量
        n_features=3, # 特征（状态）所占数组元素数量
        memory_size=MEMORY_SIZE, # 记忆库的大小
        e_greedy_increment=0.001, # 贪婪度的增量
        double_q=False, # 是否使用Double DQN
        sess=sess # tensorflow的session
    )

# 实例化Double_DQN算法类
with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, # 动作的数量
        n_features=3, # 特征（状态）所占数组元素数量
        memory_size=MEMORY_SIZE, # 记忆库的大小
        e_greedy_increment=0.001, # 贪婪度的增量
        double_q=True, # 是否使用Double DQN
        sess=sess, # tensorflow的session
        output_graph=True # 是否输出tensorboard文件
    )

sess.run(tf.global_variables_initializer()) # 初始化tensorflow的所有变量

# 控制Agent运行
def run_agent(RL):
    total_steps = 0  # 定义所有回合的总执行步数
    observation = env.reset()  # 获取初始观测状态
    while True:
        # env.render() # 显示动画效果

        action = RL.choose_action(observation)  # 选择动作
        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # 转换动作
        observation_, reward, done, info = env.step(np.array([f_action]))  # 获取执行动作后的下一个观测状态、奖励、结束标志等

        reward /= 10

        RL.store_transition(observation, action, reward, observation_)  # 将执行一步的transition存储都爱记忆中

        # 执行步数超过一定数量则开始学习
        if total_steps > MEMORY_SIZE:
            RL.learn()

        # 执行步数超过一定数量则停止训练
        if total_steps - MEMORY_SIZE > 20000:
            break

        observation = observation_  # 进入下一个观测状态
        total_steps += 1

    return RL.q

q_natural = run_agent(natural_DQN) # 获取DQN算法的Q-value数组
q_double = run_agent(double_DQN) # 获取DoubleDQN算法的Q-value数组

# 画出两种算法的Q-value变化曲线
plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q level')
plt.xlabel('training steps')
plt.grid()
plt.show()
"""
强化学习大脑：Q-learning和Sarsa
"""

import numpy as np
import pandas as pd

# 定义强化学习基类
class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # 动作空间
        self.lr = learning_rate # 学习率
        self.gamma = reward_decay # 奖励衰减
        self.epsilon = e_greedy # 贪婪度

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # 初始化q_table

    # 检查状态state是否存在于q_table中，若不存在则添加
    def check_state_exist(self, state):
        if state not in self.q_table.index: # 如果state不存在于q_table中，则添加
            self.q_table = self.q_table.append(
                pd.Series( # 一维数组
                    [0]*len(self.actions), # Series的值，初始化为0
                    index=self.q_table.columns, # Series的索引，与q_table的columns相同
                    name=state, # Series的名称，与state相同（q_table的index）
                )
            )

    # epsilon-greedy选择动作
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # 选择动作
        if np.random.rand() < self.epsilon:
            state_action = self.q_table.loc[observation, :] # 选择最佳动作
            action = np.random.choice(state_action[state_action == np.max(state_action)].index) # 如果有多个相同的最大值，随机选择一个
        else:
            action = np.random.choice(self.actions) # 随机选择动作

        return action

    # 学习更新q_table
    def learn(self, *args):
        pass


# Q-Learning算法类（off-policy）
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # 调用父类的构造函数，从而继承父类的属性和方法
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    # 学习更新q_table
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_) # 检查s_是否存在于q_table中

        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新q(s,a)

# Sarsa算法类（on-policy）
class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # 调用父类的构造函数，从而继承父类的属性和方法
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    # 学习更新q_table
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_) # 检查s_是否存在于q_table中

        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新q(s,a)

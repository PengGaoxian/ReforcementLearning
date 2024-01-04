"""
强化学习大脑：具体的强化学习算法
"""

import numpy as np
import pandas as pd

# 创建Q-Learning算法类
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 检查状态state是否存在于Q表中，若不存在则添加
    def check_state_exist(self, state):
        if state not in self.q_table.index: # 如果state不在Q表中，则增加一个一维数组，数据全0
            self.q_table = self.q_table.append(
                pd.Series( # 一维数组
                    [0]*len(self.actions), # Series的数据，全0
                    index=self.q_table.columns, # Series的索引，与Q表的列名相同
                    name=state # Series的名称，与Q表的索引相同
                    )
            )

    # epsilon-greedy选择动作
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # 选择动作
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :] # 选择最佳动作
            action = np.random.choice(state_action[state_action == np.max(state_action)].index) # 如果有多个相同的最大值，随机选择一个
        else:
            action = np.random.choice(self.actions) # 随机选择动作

        return action

    # 学习更新Q表
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_) # 检查下一个状态是否存在于Q表中

        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新q(s,a)


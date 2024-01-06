"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 260  # 奖励大于400则显示模拟窗口
RENDER = False  # 默认不显示模拟窗口

env = gym.make('CartPole-v0') # 创建CartPole-v0环境
env.seed(1)     # 设置环境的随机种子
env = env.unwrapped # 取消限制

print("动作空间：", env.action_space) # 查看这个环境中可用的action有多少个
print("状态空间：", env.observation_space) # 查看这个环境中可用的state的observation有多少个
print("状态空间的最大值：", env.observation_space.high) # 查看observation最高取值
print("状态空间的最小值：", env.observation_space.low) # 查看observation最低取值

# 实例化PolicyGradient类
RL = PolicyGradient(
    n_actions=env.action_space.n, # 动作的数量
    n_features=env.observation_space.shape[0], # 特征（状态）占数组元素的数量
    learning_rate=0.02, # 学习率
    reward_decay=0.99, # 奖励衰减率
    # output_graph=True,
)

# 运行3000个回合
for i_episode in range(3000):
    observation = env.reset() # 重置环境，开始新的回合

    while True:
        if RENDER:  # 达到显示阈值则显示模拟窗口
            env.render()

        action = RL.choose_action(observation) # 根据当前状态选择动作
        observation_, reward, done, info = env.step(action) # 执行动作，获得下一个状态、奖励、是否终止、调试信息
        RL.store_transition(observation, action, reward) # 存储这一回合的观测、动作和奖励

        if done: # 如果回合结束
            ep_rs_sum = sum(RL.ep_rs) # 计算这一回合的总奖励

            # 没有定义running_reward，直接赋值ep_rs_sum，否则计算running_reward
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            # 如果总奖励大于阈值，则显示模拟窗口
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn() # 学习，输出回合中每一步的折扣奖励作为状态价值vt

            # 在第一回合结束时绘制状态价值vt
            if i_episode == 0:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_ # 更新状态

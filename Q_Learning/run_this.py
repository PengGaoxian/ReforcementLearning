"""
控制Agent与环境的交互
"""

from maze_env import Maze
from RL_brain import QLearningTable

# 控制Agent运行的函数
def run_agent():
    for episode in range(100):
        observation = env.reset() # 初始化观测到的状态

        while True:
            env.render() # 更新画布
            action = RL.choose_action(str(observation)) # 根据当前状态选择动作
            observation_, reward, done = env.step(action) # 根据动作获取下一个状态、奖励和是否结束
            RL.learn(str(observation), action, reward, str(observation_)) # 学习更新Q表
            observation = observation_ # 更新观测状态

            if done:
                break

    # 所有回合结束后销毁环境
    print('completed all episodes')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, run_agent) # 在100ms后执行run_agent函数
    env.mainloop() # 显示迷宫
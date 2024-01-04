"""
控制Agent与环境交互的主程序
"""

from maze_env import Maze
from RL_brain import QLearningTable, SarsaTable

# 控制Agent运行的函数，参数为算法名称
def run_agent(parameter):
    for episode in range(100):
        observation = env.reset() # 初始化环境

        action = 0 # 声明action变量，用于while循环中使用

        if parameter == 'Sarsa':
            action = RL.choose_action(str(observation)) # Sarsa算法需要先选择一个动作

        while True:
            env.render() # 更新画布

            if parameter == 'QLearning':
                action = RL.choose_action(str(observation)) # QLearning算法每一步都需要选择一个动作

            observation_, reward, done = env.step(action) # 与环境交互，获得下一个状态、奖励和是否结束

            if parameter == 'Sarsa':
                action_ = RL.choose_action(str(observation_)) # Sarsa算法需要选择下一个动作

            if parameter == 'QLearning':
                RL.learn(str(observation), action, reward, str(observation_)) # QLearning算法学习更新q_table
            else:
                RL.learn(str(observation), action, reward, str(observation_), action_) # Sarsa算法学习更新q_table

            observation = observation_ # 更新状态

            if parameter == 'Sarsa':
                action = action_ # 更新动作

            # 完成一回合后退出循环
            if done:
                break

    # 跑完100个回合后销毁环境
    print('completed all episodes')
    env.destroy()

if __name__ == "__main__":
    env = Maze() # 实例化迷宫环境

    Algorithm = 'Sarsa' # 'QLearning' or 'Sarsa'
    if Algorithm == 'QLearning':
        RL = QLearningTable(actions=list(range(env.n_actions)))
    elif Algorithm == 'Sarsa':
        RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, run_agent(Algorithm)) # 在100ms后执行run_agent函数
    env.mainloop() # 显示迷宫
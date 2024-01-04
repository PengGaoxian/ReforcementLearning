from maze_env import Maze
from RL_brain import DeepQNetwork


def run_agent():
    step = 0 # 记录步数

    # 运行300个回合
    for episode in range(300):
        observation = env.reset() # 重置环境

        while True:
            env.render() # 更新画布

            action = RL.choose_action(observation) # 根据观测状态选择动作
            observation_, reward, done = env.step(action) # 执行动作，获取下一个状态的观测值、奖励、结束标志
            RL.store_transition(observation, action, reward, observation_) # 存储记忆

            if (step > 200) and (step % 5 == 0): # 超过200步后每5步学习一次
                RL.learn()

            observation = observation_ # 更新状态

            # 如果回合结束，跳出循环
            if done:
                break

            step += 1 # 步数加1

    print('completed all episodes')
    env.destroy()


if __name__ == "__main__":
    env = Maze() # 实例化环境

    RL = DeepQNetwork(env.n_actions, env.n_features, # 实例化DQN算法
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    env.after(100, run_agent) # 在100ms后执行run_agent函数
    env.mainloop() # 显示迷宫
    RL.plot_cost() # 画出cost变化曲线
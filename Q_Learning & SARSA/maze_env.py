"""
环境：用于与Agent互动，根据Agent观测到的状态state和action返回下一个状态state_和奖励reward
"""

import numpy as np
import time
import tkinter as tk

# 定义迷宫的大小：单元格大小设置为40pix，x轴单元格个数：4，y轴单元格个数：4
UNIT = 40   # 单元格边长
MAZE_H = 4  # Y轴单元格个数
MAZE_W = 4  # X轴单元格个数

# 定义起点、障碍、终点的坐标
origin_grid = np.array([0, 0]) # 起点的网格坐标
hells_grid = np.array([[2,1], [1,2], [3,0]]) # 障碍的网格坐标
oval_grid = np.array([2,2]) # 终点的网格坐标

#  定义迷宫Maze类
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()

        self.action_space = ['u', 'd', 'l', 'r'] # 定义动作空间
        self.n_actions = len(self.action_space) # 计算动作数量

        self.title('maze') # 设置标题
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT)) # 设置窗口大小

        self._build_maze() # 构建迷宫

    def _build_maze(self):
        # 创建画布
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # 创建网格（横线和竖线）
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 创建起点的中心点坐标
        origin_center = np.array([20, 20])

        # 创建陷阱（黑色方块）
        self.hells = np.zeros(len(hells_grid), int)
        for i in range(len(hells_grid)):
            self.hells_center = origin_center + hells_grid[i] * UNIT
            self.hells[i] = self.canvas.create_rectangle(
                self.hells_center[1] - 15, self.hells_center[0] - 15,
                self.hells_center[1] + 15, self.hells_center[0] + 15,
                fill='black')

        # 创建终点（黄色圆形）
        oval_center = origin_center + oval_grid * UNIT
        self.oval = self.canvas.create_oval(
            oval_center[1] - 15, oval_center[0] - 15,
            oval_center[1] + 15, oval_center[0] + 15,
            fill='yellow')

        # 创建探索者（红色方块）
        rect_center = origin_center + origin_grid * UNIT
        self.rect = self.canvas.create_rectangle(
            rect_center[1] - 15, rect_center[0] - 15,
            rect_center[1] + 15, rect_center[0] + 15,
            fill='red')

        # 显示画布
        self.canvas.pack()

    # 重置环境
    def reset(self):
        self.update() # 更新画布
        time.sleep(0.5)
        self.canvas.delete(self.rect) # 删除探索者（红色方块）
        origin_center = np.array([20, 20])
        self.rect = self.canvas.create_rectangle( # 重新创建探索者（红色方块）
            origin_center[0] - 15, origin_center[1] - 15,
            origin_center[0] + 15, origin_center[1] + 15,
            fill='red')

        return self.canvas.coords(self.rect) # 返回探索者的坐标

    # 执行动作后，返回下一个状态、奖励和是否结束
    def step(self, action):
        # 计算下一个状态
        s = self.canvas.coords(self.rect) # 获取探索者的坐标
        base_action = np.array([0, 0]) # 定义位移量
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # 根据位移量base_action移动探索者
        s_ = self.canvas.coords(self.rect)  # 获取移动后的探索者坐标（左上角和右下角坐标）

        # 计算奖励和是否结束
        hells_coord = [] # 存储障碍物的坐标
        for i in range(len(self.hells)):
            hells_coord.append(self.canvas.coords(self.hells[i]))

        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif s_ in hells_coord:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return s_, reward, done

    # 更新画布
    def render(self):
        time.sleep(0.1)
        self.update()

# 运行智能体与环境交互
def run_agent():
    for t in range(10): # 跑10个回合
        s = env.reset()
        while True:
            env.render()
            a = 1 # down
            s_, r, done = env.step(a)
            if done:
                break

    # 跑完10个回合后销毁环境
    print('completed all episodes')
    env.destroy()

if __name__ == '__main__':
    env = Maze() # 实例化迷宫环境
    env.after(100, run_agent) # 在100ms后执行run函数
    env.mainloop() # 显示迷宫
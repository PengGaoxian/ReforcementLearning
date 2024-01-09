"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt


GAME = 'CartPole-v0' # 游戏名称
OUTPUT_GRAPH = True # 是否输出tensorboard文件
LOG_DIR = './log' # tensorboard文件保存路径
N_WORKERS = multiprocessing.cpu_count() # CPU核数
MAX_GLOBAL_EP = 1000 # 训练的最大回合数
GLOBAL_NET_SCOPE = 'Global_Net' # 全局网络名称
UPDATE_GLOBAL_ITER = 10 # 更新全局网络的迭代次数
GAMMA = 0.9 # 奖励衰减率
ENTROPY_BETA = 0.001 # 熵的权重
LR_A = 0.001    # Actor的学习率
LR_C = 0.001    # Critic的学习率
GLOBAL_RUNNING_R = [] # 存储每一回合的总奖励
GLOBAL_EP = 0 # 回合数

env = gym.make(GAME) # 创建游戏环境
N_S = env.observation_space.shape[0] # 状态空间的维度
N_A = env.action_space.n # 动作空间的维度


# A3C的网络结构：分为全局网络和局部网络，全局网络不需要训练，仅训练局部网络。局部网络又分为Actor和Critic网络
class ACNet(object):
    def __init__(self, scope, globalAC=None): # globalAC指的是全局网络，用于同步参数
        # 创建全局Actor-Critic网络，并获取全局网络的参数
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                # 创建全局Actor网络和全局Critic网络,并获取参数
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        # 创建局部网络
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S') # UPDATA_GLOBAL_ITER=10个状态
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A') # UPDATA_GLOBAL_ITER=10个动作
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget') # UPDATA_GLOBAL_ITER=10个目标价值

                # 创建局部Actor网络和局部Critic网络，并获取动作概率和V(s)，以及网络参数
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error') # 计算TD误差：v_target - v

                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td)) # 计算TD误差的平方作为Critic的损失

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True) # 计算概率的对数
                    exp_v = log_prob * tf.stop_gradient(td) # 计算V(s)，构建一个节点td，当计算梯度时，不要计算这个节点
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5), axis=1, keep_dims=True)  # 计算熵，用于探索
                    self.exp_v = ENTROPY_BETA * entropy + exp_v # 计算V(s) + 熵
                    self.a_loss = tf.reduce_mean(-self.exp_v) # 计算Actor的损失

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params) # 计算Actor的梯度
                    self.c_grads = tf.gradients(self.c_loss, self.c_params) # 计算Critic的梯度

            # 局部网络与全局网络同步
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)] # 更新局部的Actor的参数
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)] # 更新局部的Critic的参数
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params)) # 更新全局的Actor的参数
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params)) # 更新全局的Critic的参数

    # 创建Actor和Critic网络
    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1) # 权重初始化

        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la') # 创建Actor的隐藏层
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap') # 创建Actor的输出层，输出动作的概率

        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc') # 创建Critic的隐藏层
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # 创建Critic的输出层，输出状态的价值

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor') # 获取Actor的参数
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic') # 获取Critic的参数

        return a_prob, v, a_params, c_params

    # 更新全局网络
    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    # 更新局部网络
    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    # 选择动作
    def choose_action(self, s):
        # 选择动作的概率
        prob_weights = SESS.run(self.a_prob,
                                feed_dict={self.s: s[np.newaxis, :]})
        # 根据概率选择动作
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())
        return action


# 创建Worker
class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped # 创建游戏环境，取消限制
        self.name = name # Worker的名称
        self.AC = ACNet(name, globalAC) # 创建Actor-Critic网络

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1 # 总步数
        buffer_s = [] # 定义存储状态的数组
        buffer_a = [] # 定义存储动作的数组
        buffer_r = [] # 定义存储奖励的数组

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset() # 初始化状态
            ep_r = 0 # 回合的总奖励
            while True:
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s) # 选择动作
                s_, r, done, info = self.env.step(a) # 执行动作

                if done:
                    r = -5

                ep_r += r # 计算回合的总奖励

                buffer_s.append(s) # 存储状态
                buffer_a.append(a) # 存储动作
                buffer_r.append(r) # 存储奖励

                # 执行步数达到更新步数或者回合结束时，更新全局网络
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0 # 回合结束时，状态的价值为0
                    else:
                        v_s_ = SESS.run(self.AC.v, feed_dict={self.AC.s: s_[np.newaxis, :]})[0, 0] # 计算状态的价值

                    buffer_v_target = [] # 定义存储目标价值的数组

                    for r in buffer_r[::-1]:    # 逆序遍历buffer_r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)

                    buffer_v_target.reverse() # 反转数组

                    buffer_s = np.vstack(buffer_s)
                    buffer_a = np.array(buffer_a)
                    buffer_v_target = np.vstack(buffer_v_target)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s = []
                    buffer_a = []
                    buffer_r = []

                    self.AC.pull_global()

                s = s_
                total_step += 1

                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

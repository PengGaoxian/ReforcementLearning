"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time


np.random.seed(1) # 设置numpy的随机数种子
tf.set_random_seed(1) # 设置tensorflow的随机数种子

#####################  hyper parameters  ####################

MAX_EPISODES = 200 # 运行的最大回合数
MAX_EP_STEPS = 200 # 每回合最大的运行步数
LR_A = 0.001    # Actor的学习率
LR_C = 0.001    # Critic的学习率
GAMMA = 0.9     # 奖励衰减率

# 两种target网络的更新方式
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]

MEMORY_CAPACITY = 10000 # 记忆库的容量
BATCH_SIZE = 32 # 每次更新时从记忆库中取出的样本数

RENDER = False # 是否显示游戏画面
OUTPUT_GRAPH = True # 是否输出tensorboard文件
ENV_NAME = 'Pendulum-v0' # 游戏名称

###############################  Actor  ####################################
class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim # 动作的维度
        self.action_bound = action_bound # 动作的范围
        self.lr = learning_rate # 学习率
        self.replacement = replacement # target网络的更新方式
        self.t_replace_counter = 0 # target网络的更新计数器

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a_ for critic network
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net') # eval_net的参数
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net') # target_net的参数

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            # 将eval_net的参数赋值给target_net
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            # soft方式更新target_net的参数
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    # 构建Actor的神经网络
    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3) # 权重的初始化
            init_b = tf.constant_initializer(0.1) # 偏置的初始化
            # 第一个全连接层
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                # 第二个全连接层，输出动作
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # 缩放动作到[-action_bound, +action_bound]区间

        return scaled_a

    # 学习更新Actor网络
    def learn(self, s):   # batch update
        # 通过Actor的eval_net计算出动作
        self.sess.run(self.train_op,
                      feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace) # soft方式更新target_net的参数
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace) # hard方式更新target_net的参数

            self.t_replace_counter += 1 # 更新计数器

    def choose_action(self, s):
        s = s[np.newaxis, :]    # 单个状态
        # 通过Actor的eval_net计算出单个状态下的动作
        return self.sess.run(self.a,
                             feed_dict={S: s})[0]

    # 将Actor的梯度传递给Critic网络
    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # e_params是要求的参数;
            # a_grads是从critic中传过来的梯度
            # 这行代码就是求actor中参数的梯度公式
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################
class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim # 状态的维度
        self.a_dim = action_dim # 动作的维度
        self.lr = learning_rate # Critic的学习率
        self.gamma = gamma # 奖励衰减率
        self.replacement = replacement # target网络的更新方式

        with tf.variable_scope('Critic'):
            # stop critic update flows to actor
            self.a = tf.stop_gradient(a)
            # 通过Critic的eval_net计算出Q值，# Input (s, a), output q
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # 通过Critic的target_net计算出下一个动作的Q值，Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net') # eval_net的参数
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net') # target_net的参数

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_ # 计算目标Q值

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q)) # TD误差

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss) # Critic的优化器

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # q对a求导，用于传入actor网络对参数求导，tensor of gradients of each sample (None, a_dim)

        # 根据target网络的更新方式，选择不同的更新方式
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    # 构建Critic的神经网络
    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1) # 权重的初始化
            init_b = tf.constant_initializer(0.1) # 偏置的初始化

            with tf.variable_scope('l1'):
                n_l1 = 30 # 第一层神经元的个数
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable) # 第一层状态的权重（并列）
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable) # 第一层动作的权重（并列）
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable) # 第一层的偏置
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1) # 第一层的输出

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)

        return q

    def learn(self, s, a, r, s_):
        # 通过Critic的eval_net计算出Q值
        self.sess.run(self.train_op,
                      feed_dict={S: s, self.a: a, R: r, S_: s_})

        # 根据target网络的更新方式，选择不同的更新方式
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity # 记忆库的容量
        self.data = np.zeros((capacity, dims)) # 记忆库
        self.pointer = 0 # 记忆库的指针

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_)) # 将状态、动作、奖励、下一个状态拼接起来
        index = self.pointer % self.capacity  # 生成记忆库的索引，如果超过记忆库的容量，则覆盖之前的记忆
        self.data[index, :] = transition # 存储记忆
        self.pointer += 1 # 记忆库的指针加1

    # 从记忆库中随机取出n个记忆
    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled' # 如果记忆库没有被填满，则报错
        indices = np.random.choice(self.capacity, size=n) # 随机生成n个索引
        return self.data[indices, :] # 返回n个记忆


env = gym.make(ENV_NAME) # 创建游戏环境
env = env.unwrapped # 取消限制
env.seed(1) # 设置游戏的随机数种子

state_dim = env.observation_space.shape[0] # 状态的维度
action_dim = env.action_space.shape[0] # 动作的维度
action_bound = env.action_space.high # 动作的范围

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()

# 实例化Actor，创建Actor的eval_net和target_net
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
# 实例化Critic，创建Critic的eval_net和target_net
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
# 将critic的梯度传递给actor，用于actor的参数更新
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer()) # 初始化所有变量

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1) # 实例化记忆库

# 输出tensorboard文件
if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

var = 3  # 控制探索的程度

t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset() # 重置游戏环境
    ep_reward = 0 # 每回合的奖励

    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        # Add exploration noise
        a = actor.choose_action(s) # 通过Actor的eval_net计算出动作
        a = np.clip(np.random.normal(a, var), -2, 2) # 添加探索噪声
        s_, r, done, info = env.step(a) # 执行动作，获取下一个状态、奖励、是否结束的标志位

        M.store_transition(s, a, r / 10, s_) # 存储记忆

        if M.pointer > MEMORY_CAPACITY:
            var *= .9995    # 探索的程度逐渐减小
            b_M = M.sample(BATCH_SIZE) # 从记忆库中随机取出BATCH_SIZE个记忆
            b_s = b_M[:, :state_dim] # 取出状态
            b_a = b_M[:, state_dim: state_dim + action_dim] # 取出动作
            b_r = b_M[:, -state_dim - 1: -state_dim] # 取出奖励
            b_s_ = b_M[:, -state_dim:] # 取出下一个状态

            critic.learn(b_s, b_a, b_r, b_s_) # 学习更新Critic网络
            actor.learn(b_s) # 学习更新Actor网络

        s = s_ # 更新状态
        ep_reward += r # 更新回合奖励

        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:
                RENDER = True
            break

print('Running time: ', time.time()-t1)
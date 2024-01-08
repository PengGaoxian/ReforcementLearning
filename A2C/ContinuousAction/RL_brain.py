"""
Actor-Critic with continuous action using TD-error as the Advantage, Reinforcement Learning.

The Pendulum example (based on https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb)

Cannot converge!!! oscillate!!!

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow r1.3
gym 0.8.0
"""

import tensorflow as tf
import numpy as np

np.random.seed(2) # 设置numpy的随机数种子
tf.set_random_seed(2)  # 设置tensorflow的随机数种子


# Actor类
class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001):
        self.sess = sess # 接收tensorflow会话

        self.s = tf.placeholder(tf.float32, [1, n_features], "state") # 状态
        self.a = tf.placeholder(tf.float32, None, name="act") # 动作
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        # 第一层全连接层
        l1 = tf.layers.dense(
            inputs=self.s, # 输入状态
            units=30,  # 神经元个数
            activation=tf.nn.relu, # 激活函数
            kernel_initializer=tf.random_normal_initializer(0., .1),  # 权重初始化
            bias_initializer=tf.constant_initializer(0.1),  # 偏置初始化
            name='l1' # 层名称
        )

        # 第二层全连接层（并行）：输出动作的均值
        mu = tf.layers.dense(
            inputs=l1, # 输入
            units=1,  # 神经元个数
            activation=tf.nn.tanh, # 激活函数
            kernel_initializer=tf.random_normal_initializer(0., .1),  # 权重初始化
            bias_initializer=tf.constant_initializer(0.1),  # 偏置初始化
            name='mu' # 层名称
        )

        # 第二层全连接层（并行）：输出动作的方差
        sigma = tf.layers.dense(
            inputs=l1, # 输入
            units=1,  # 神经元个数
            activation=tf.nn.softplus,  # 激活函数
            kernel_initializer=tf.random_normal_initializer(0., .1),  # 权重初始化
            bias_initializer=tf.constant_initializer(1.),  # 偏置初始化
            name='sigma' # 层名称
        )

        global_step = tf.Variable(0, trainable=False) # 创建一个全局计数器，用于记录训练步数
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)

        self.mu = tf.squeeze(mu*2) # 均值
        self.sigma = tf.squeeze(sigma+0.1) # 方差
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma) # 根据均值和方差创建正态分布

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1]) # 根据正态分布采样一个动作

        # 计算动作的概率
        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)  # 动作概率的对数
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) 计算优势函数A(s,a)=V(s)-Q(s,a)
            self.exp_v += 0.01*self.normal_dist.entropy() # 为了鼓励探索，添加交叉熵损失

        # 训练网络参数
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step) # 最大化exp_v，global_step用于记录训练步数

    # 学习更新网络参数，返回V(s)
    def learn(self, s, a, td):
        s = s[np.newaxis, :] # 将状态转换为矩阵，方便输入网络
        # 将状态、动作、优势函数输入网络，计算V(s)
        _, exp_v = self.sess.run([self.train_op, self.exp_v],
                                 feed_dict={self.s: s, # 输入状态
                                            self.a: a, # 输入动作
                                            self.td_error: td}) # 输入优势函数
        # 返回V(s)
        return exp_v

    # 根据状态选择动作
    def choose_action(self, s):
        s = s[np.newaxis, :] # 将状态转换为矩阵，方便输入网络
        # 根据状态输入网络，采样一个动作
        action = self.sess.run(self.action,
                                 feed_dict={self.s: s})
        return action


# Critic类
class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, gamma=0.9):
        self.sess = sess # 接收tensorflow会话
        self.s = tf.placeholder(tf.float32, [1, n_features], "state") # 状态
        self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next") # 下一个状态的V(s)
        self.r = tf.placeholder(tf.float32, name='r') # 奖励

        with tf.variable_scope('Critic'):
            # 第一层全连接层
            l1 = tf.layers.dense(
                inputs=self.s, # 输入状态
                units=30,  # 神经元个数
                activation=tf.nn.relu, # 激活函数
                kernel_initializer=tf.random_normal_initializer(0., .1),  # 权重初始化
                bias_initializer=tf.constant_initializer(0.1),  # 偏置初始化
                name='l1' # 层名称
            )

            self.v = tf.layers.dense(
                inputs=l1, # 输入
                units=1,  # 输出神经元个数
                activation=None, # 激活函数
                kernel_initializer=tf.random_normal_initializer(0., .1),  # 权重初始化
                bias_initializer=tf.constant_initializer(0.1),  # 偏置初始化
                name='V' # 层名称
            )

        # 计算TD_error
        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + gamma * self.v_ - self.v) # TD_error = (r+gamma*v_) - v
            self.loss = tf.square(self.td_error)    # loss为TD_error的平方
        # 训练网络参数
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    # 学习更新网络参数，返回TD_error
    def learn(self, s, r, s_):
        s = s[np.newaxis, :] # 将状态转换为矩阵，方便输入网络
        s_ = s_[np.newaxis, :] # 将下一个状态转换为矩阵，方便输入网络

        # 计算下一个状态的V(s_)
        v_ = self.sess.run(self.v,
                           feed_dict={self.s: s_}) # 输入下一个状态
        # 计算TD_error
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          feed_dict={self.s: s, # 输入状态
                                                     self.v_: v_, # 下一个状态的V(s_)
                                                     self.r: r}) # 奖励
        return td_error

"""
强化学习大脑：A2C算法

Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(2) # 设置numpy的随机数种子
tf.set_random_seed(2)  # 设置tensorflow的随机数种子

# Actor类
class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state") # 状态
        self.a = tf.placeholder(tf.int32, None, "act") # 动作
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        # 创建Actor神经网络
        with tf.variable_scope('Actor'):
            # 第一层全连接层
            l1 = tf.layers.dense(
                inputs=self.s, # 输入状态
                units=20,    # 神经元个数
                activation=tf.nn.relu, # 激活函数
                kernel_initializer=tf.random_normal_initializer(0., .1),    # 权重初始化
                bias_initializer=tf.constant_initializer(0.1),  # 偏置初始化
                name='l1' # 层名称
            )

            # 第二层全连接层
            self.acts_prob = tf.layers.dense(
                inputs=l1, # 输入
                units=n_actions,  # 输出动作
                activation=tf.nn.softmax,   # 激活函数，输出动作的概率
                kernel_initializer=tf.random_normal_initializer(0., .1),  # 权重初始化
                bias_initializer=tf.constant_initializer(0.1),  # 偏置初始化
                name='acts_prob' # 层名称
            )

        # AC算法是通过蒙特卡洛方法来计算V(s)的，这里A2C用优势函数A(s,a)=V(s)-Q(s,a)来替代V(s)
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a]) # 将动作概率转换为对数
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # 优势函数(TD_error)替代V(s)

        # 训练网络参数
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    # 学习更新网络参数
    def learn(self, s, a, td):
        s = s[np.newaxis, :] # 将状态转换为矩阵，方便输入网络

        # 将状态、动作、优势函数输入网络，计算V(s)
        _, exp_v = self.sess.run([self.train_op, self.exp_v],
                                 feed_dict={self.s: s,
                                            self.a: a,
                                            self.td_error: td})
        return exp_v # 返回V(s)

    # 根据状态选择动作
    def choose_action(self, s):
        s = s[np.newaxis, :] # 将状态转换为矩阵，方便输入网络
        # 根据状态输入网络，得到所有动作的概率
        probs = self.sess.run(self.acts_prob,
                              feed_dict={self.s: s})
        # 根据概率选择动作
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


# Critic类
class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, gamma=0.9):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state") # 状态
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next") # 下一个状态的V(s)
        self.r = tf.placeholder(tf.float32, None, 'r') # 奖励

        # 创建Critic神经网络
        with tf.variable_scope('Critic'):
            # 第一层全连接层
            l1 = tf.layers.dense(
                inputs=self.s, # 输入状态
                units=20,  # 神经元个数
                activation=tf.nn.relu,  # 激活函数
                kernel_initializer=tf.random_normal_initializer(0., .1),  # 权重初始化
                bias_initializer=tf.constant_initializer(0.1),  # 偏置初始化
                name='l1' # 层名称
            )

            # 第二层全连接层
            self.v = tf.layers.dense(
                inputs=l1, # 输入
                units=1,  # 神经元个数
                activation=None, # 激活函数
                kernel_initializer=tf.random_normal_initializer(0., .1),  # 权重初始化
                bias_initializer=tf.constant_initializer(0.1),  # 偏置初始化
                name='V' # 层名称
            )

        # 计算TD_error
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + gamma * self.v_ - self.v # TD_error = A(s,a) = Q(s,a) - V(s) = r + gamma * V(s+1) - V(s)
            self.loss = tf.square(self.td_error) # TD_error的平方
        # 训练网络参数
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    # 学习更新网络参数
    def learn(self, s, r, s_):
        s = s[np.newaxis, :] # 将状态转换为矩阵，方便输入网络
        s_ = s_[np.newaxis, :] # 将下一个状态转换为矩阵，方便输入网络

        v_ = self.sess.run(self.v,  # 根据下一个状态计算V(s+1)
                           feed_dict={self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], # 根据状态、奖励、下一个状态更新网络参数，计算TD_error
                           feed_dict={self.s: s,
                                      self.v_: v_, self.r: r})
        # 返回TD_error = A(s,a) = Q(s,a) - V(s)
        return td_error

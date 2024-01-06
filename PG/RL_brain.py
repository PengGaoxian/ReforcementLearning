"""
强化学习大脑：PolicyGradient算法

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1) # 设置numpy的随机数种子
tf.set_random_seed(1) # 设置tensorflow的随机数种子

# 定义Policy_Gradient算法类
class PolicyGradient:
    def __init__(
            self,
            n_actions, # 动作的数量
            n_features, # 特征（状态）占数组元素的数量
            learning_rate=0.01, # 学习率
            reward_decay=0.95, # 奖励衰减率
            output_graph=False, # 是否输出计算图
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs = [] # 存储每一回合的观测
        self.ep_as = [] # 存储每一回合的动作
        self.ep_rs = [] # 存储每一回合的奖励

        self._build_net() # 构建神经网络

        self.sess = tf.Session() # 创建会话

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph) # 输出计算图

        self.sess.run(tf.global_variables_initializer()) # 初始化tensorflow的所有变量

    # 构建神经网络
    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations") # 创建观测的占位符
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num") # 创建动作的占位符
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value") # 创建动作价值的占位符
        # 第一层全连接层
        layer = tf.layers.dense(
            inputs=self.tf_obs, # 输入：观测状态
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # 第二层全连接层
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # 第三层softmax输出层（将每个动作对应的值转换为概率）
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            # 最大化总reward，即(log_p * R)， 相当于最小化 -(log_p * R)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1) # 获取一个回合中每个动作的概率
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # 将每个动作的概率乘以对应的奖励，再求平均值，作为loss（本质上没有loss，因为没有目标）
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss) # 使用Adam优化器最小化loss

    # 选择动作
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob,  # 根据观测状态observation获取每个动作的概率
                                     feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel()) # 根据概率选择动作
        return action

    # 存储一个回合的观测、动作和奖励
    def store_transition(self, s, a, r):
        self.ep_obs.append(s) # 存储观测
        self.ep_as.append(a) # 存储动作
        self.ep_rs.append(r) # 存储奖励

    # 折扣并标准化回合中每一步的奖励
    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs) # 创建一个和奖励数组ep_rs形状相同的数组，用于存储折扣后的奖励
        running_add = 0 # 用于存储每个回合的折扣后的奖励

        # 从回合最后一步开始，向前计算的每一步的折扣奖励
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t] # 计算回合中某一步的折扣奖励
            discounted_ep_rs[t] = running_add # 存储回合中某一步的折扣奖励

        # 标准化回合奖励
        discounted_ep_rs -= np.mean(discounted_ep_rs) # 减去平均值
        discounted_ep_rs /= np.std(discounted_ep_rs) # 除以标准差

        return discounted_ep_rs

    # 学习更新神经网络参数
    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards() # 折扣并标准化回合奖励

        # 训练神经网络
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # 回合中的所有观测状态，shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # 回合中的所有动作，shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # 回合中每个动作对应的标准化后的折扣奖励，shape=[None, ]
        })

        self.ep_obs = [] # 清空本回合的观测
        self.ep_as = [] # 清空本回合的动作
        self.ep_rs = [] # 清空本回合的奖励

        # 返回回合中每一步的折扣奖励
        return discounted_ep_rs_norm
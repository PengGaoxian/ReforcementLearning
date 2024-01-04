"""
强化学习大脑：Double DQN

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1) # 设置numpy的随机种子
tf.set_random_seed(1) # 设置tensorflow的随机种子

# 定义Double DQN的算法类
class DoubleDQN:
    def __init__(
            self,
            n_actions, # 动作的数量
            n_features, # 特征（状态）所占数组元素数量
            learning_rate=0.005, # 学习率
            reward_decay=0.9, # 奖励衰减率
            e_greedy=0.9, # 贪婪度
            replace_target_iter=200, # 间隔多少次学习更新一次target网络的参数
            memory_size=3000, # 记忆库的大小
            batch_size=32, # 每次学习时从记忆库中取出的样本数量
            e_greedy_increment=None, # 贪婪度的增量
            output_graph=False, # 是否输出tensorboard文件
            double_q=True, # 是否使用Double DQN
            sess=None, # 是否使用已有的tensorflow session
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.double_q = double_q    # decide to use double q or not
        self.learn_step_counter = 0 # 记录学习了多少次
        self.memory = np.zeros((self.memory_size, n_features*2+2)) # 初始化记忆库，存储[s, a, r, s_]

        self._build_net() # 创建神经网络

        t_params = tf.get_collection('target_net_params') # 获取target网络的参数
        e_params = tf.get_collection('eval_net_params') # 获取eval网络的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # 更新target网络的参数

        # 没有sess则创建一个新的sess，并初始化所有参数；有sess则使用已有的sess
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        # 是否输出tensorboard文件
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = [] # 记录每次学习的损失，用于最后画出cost变化曲线

    # 创建神经网络
    def _build_net(self):
        # 定义网络结构
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            # 第一层神经网络
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names) # 权重
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names) # 偏置
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1) # 计算输出

            # 第二层神经网络
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names) # 权重
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names) # 偏置
                out = tf.matmul(l1, w2) + b2 # 计算输出

            return out

        w_initializer = tf.random_normal_initializer(0., 0.3) # 权重的初始化器
        b_initializer = tf.constant_initializer(0.1) # 偏置的初始化器
        n_l1 = 20 # 第一层神经网络的神经元数量

        # eval网络
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # eval网络的输入

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES] # eval网络参数的集合
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer) # eval网络的输出

        # target网络
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') # target网络的输入
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES] # target网络参数的集合
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer) # target网络的输出


        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 损失的输入
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval)) # 计算loss
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss) # loss优化器

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0 # 初始化记忆库的计数器

        transition = np.hstack((s, [a, r], s_)) # 将[s, a, r, s_]合并成一个数组
        index = self.memory_counter % self.memory_size # 计算存储位置
        self.memory[index, :] = transition # 存储transition
        self.memory_counter += 1 # 计数器加1

    # 选择动作
    def choose_action(self, observation):
        observation = observation[np.newaxis, :] # 将observation转换成1*n_features的矩阵
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation}) # 计算q_eval
        action = np.argmax(actions_value) # 选择q_eval中能获得最大价值的动作a

        # 如果算法类中没有属性q，则创建一个列表q和变量running_q
        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value) # 计算running_q
        self.q.append(self.running_q) # 将running_q存入列表q中

        # 以1-epsilon的概率随机选择动作
        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)

        return action

    # 学习更新网络参数
    def learn(self):
        # 每隔replace_target_iter次学习，更新一次target网络的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 从记忆库中随机取出batch_size个样本
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :] # 从记忆库中取出batch_memory

        # 计算q_next(s_,a)和q_eval(s_,a)
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # target网络的输入
                       self.s: batch_memory[:, -self.n_features:]})    # eval网络的输入
        # 计算q_eval(s,a)
        q_eval = self.sess.run(
            self.q_eval,
            feed_dict={self.s: batch_memory[:, :self.n_features]}) # eval网络的输入

        q_target = q_eval.copy() # 用q_eval来初始化q_target

        batch_index = np.arange(self.batch_size, dtype=np.int32) # 创建一个整数数组，从0到self.batch_size-1
        eval_act_index = batch_memory[:, self.n_features].astype(int) # 从batch_memory中取出动作a
        reward = batch_memory[:, self.n_features + 1] # 从batch_memory中取出奖励r

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # 获取q_eval(s_)中能获得最大价值的动作a
            selected_q_next = q_next[batch_index, max_act4next]  # 计算q_next(s_,a)
        else:
            selected_q_next = np.max(q_next, axis=1)    # 获取q_next(s_)中能获得最大价值的动作a

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next # 计算q_target

        # 训练eval网络
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={self.s: batch_memory[:, :self.n_features],
                       self.q_target: q_target})

        self.cost_his.append(cost) # 记录每次学习的损失

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max # 更新贪婪度
        self.learn_step_counter += 1 # 学习次数加1
"""
强化学习大脑：Deep Q Network
"""

import numpy as np
import tensorflow as tf

# 设置numpy和tensorflow的随机种子
np.random.seed(1)
tf.set_random_seed(1)

# 创建DeepQNetwork类（off-policy）
class DeepQNetwork:
    def __init__(
            self,
            n_actions, # 动作数量
            n_features, # 特征（状态）所占的数组元素数量
            learning_rate=0.01, # 学习率
            reward_decay=0.9, # 奖励衰减率
            e_greedy=0.9, # 贪婪度
            replace_target_iter=300, # 间隔多少次学习更新一次target网络的参数
            memory_size=500, # 记忆库大小
            batch_size=32, # 批次数量
            e_greedy_increment=None, # 贪婪度增量
            output_graph=False, # 是否输出tensorboard文件
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

        self.learn_step_counter = 0 # 学习次数

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2)) # 初始化记忆变量为全0，存储[s,a,r,s]

        self._build_net() # 构建神经网络

        t_params = tf.get_collection('target_net_params') # 提取target网络的参数
        e_params = tf.get_collection('eval_net_params') # 提取eval网络的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # 更新target网络的参数

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer()) # 初始化tensorflow的所有变量
        self.cost_his = [] # 记录所有的cost变化，用于最后画出cost变化曲线

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s') # 定义eval网络的输入
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # 计算loss时的输入

        w_initializer = tf.random_normal_initializer(0., 0.3)  # 权重的初始化
        b_initializer = tf.constant_initializer(0.1)  # 偏置的初始化

        # 构建eval神经网络
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]  # eval网络的参数集合
            n_l1 = 10  # 第一层的神经元数量
            # w_initializer = tf.random_normal_initializer(0., 0.3)  # 权重的初始化器
            # b_initializer = tf.constant_initializer(0.1)  # 偏置的初始化器

            # eval网络的第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names) # 创建第一层的权重
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names) # 创建第一层的偏置
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1) # 计算第一层的输出

            # eval网络的第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names) # 创建第二层的权重
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names) # 创建第二层的偏置
                self.q_eval = tf.matmul(l1, w2) + b2 # 计算第二层的输出

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval)) # 计算loss
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss) # loss优化器

        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') # target网络的输入
        # 构建target神经网络
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES] # target网络的参数集合

            # target网络的第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names) # 创建第一层的权重
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names) # 创建第一层的偏置
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1) # 计算第一层的输出

            # target网络的第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names) # 创建第二层的权重
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names) # 创建第二层的偏置
                self.q_next = tf.matmul(l1, w2) + b2 # 计算第二层的输出

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0 # 初始化记忆库的计数器

        transition = np.hstack((s, [a, r], s_)) # 将[s,a,r,s_]合并为一个数组

        index = self.memory_counter % self.memory_size # 计算记忆存储的位置
        self.memory[index, :] = transition # 存储记忆

        self.memory_counter += 1 # 计数器加1

    # 选择动作
    def choose_action(self, observation):
        observation = observation[np.newaxis, :] # 将observation转换为1行n_features列的矩阵

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation}) # 通过eval网络计算所有动作的值
            action = np.argmax(actions_value) # 选择值最大的动作
        else:
            action = np.random.randint(0, self.n_actions) # 随机选择一个动作

        return action

    # 学习更新网络参数
    def learn(self):
        # 每隔replace_target_iter次学习更新一次target网络的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 随机抽取batch_size个记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :] # 抽取记忆

        # 计算q_next和q_eval
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features],
            })

        q_target = q_eval.copy() # 用q_eval的值初始化q_target

        batch_index = np.arange(self.batch_size, dtype=np.int32) # 构建batch_index，从0到batch_size-1的数组
        eval_act_index = batch_memory[:, self.n_features].astype(int) # 获取batch_memory中的动作
        reward = batch_memory[:, self.n_features + 1] # 获取batch_memory中的奖励

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) # 更新q_target

        # 训练eval网络
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={self.s: batch_memory[:, :self.n_features],
                       self.q_target: q_target})

        self.cost_his.append(cost) # 记录每一次学习的cost变化

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max # 更新贪婪度
        self.learn_step_counter += 1 # 学习次数加1

    # 画出cost变化曲线
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
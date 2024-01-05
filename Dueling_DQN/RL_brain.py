"""
强化学习大脑：Prior_Replay_DQN

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1) # 设置numpy的随机种子
tf.set_random_seed(1) # 设置tensorflow的随机种子


"""
Tree structure and array storage:

Tree index:
     0         -> storing priority sum
    / \
  1     2
 / \   / \
3   4 5   6    -> storing priority for transitions

Array type for storing:
[0,1,2,3,4,5,6]
"""

# 实现一个二叉树，用于存储优先级和数据
class SumTree(object):
    data_pointer = 0 # transition在data中的索引

    def __init__(self, capacity):
        self.capacity = capacity  # 定义transition的存储容量(size=4)
        self.data = np.zeros(capacity, dtype=object)  # 定义存储transitions的变量data
        self.tree = np.zeros(2 * capacity - 1) # 定义二叉树大小(size=7)

    # 添加transition及其优先级p到二叉树中
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1 # 计算transition的优先级在二叉树中的索引(3)
        self.data[self.data_pointer] = data  # 更新data中的transition
        self.update(tree_idx, p)  # 更新tree中的的优先级p

        self.data_pointer += 1 # 更新transition在data中的索引
        if self.data_pointer >= self.capacity:  # 在data中循环更新transition
            self.data_pointer = 0

    # 更新二叉树中的tree节点的优先级
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx] # 计算优先级变化
        self.tree[tree_idx] = p # 更新优先级
        while tree_idx != 0:    # 逐层遍历父节点，更新优先级之和
            tree_idx = (tree_idx - 1) // 2 # 计算父节点索引(1)
            self.tree[tree_idx] += change # 更新父节点优先级

    # 根据给定值v获取叶节点索引、优先级和对应的transition
    def get_leaf(self, v):
        parent_idx = 0 # 从根节点开始
        while True:     # 遍历子节点寻找叶节点
            cl_idx = 2 * parent_idx + 1 # 计算左子节点索引
            cr_idx = cl_idx + 1 # 计算右子节点索引

            if cl_idx >= len(self.tree): # 如果左子节点索引大于tree的容量，则返回父节点索引、父节点优先级和父节点数据
                leaf_idx = parent_idx
                break
            else: # 如果左子节点索引小于tree的容量，则继续计算，更新父节点索引
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1 # 计算transition在data中的索引

        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    # 获取根节点优先级（所有子节点优先级之和）
    @property
    def total_p(self):
        return self.tree[0]  # the root


# 存储( s, a, r, s_ )到二叉树中
class Memory(object):
    epsilon = 0.01  # TD error的最小值，避免优先级为0
    alpha = 0.6  # [0~1] 将TD error转化为优先级p的参数
    beta = 0.4  # [0~1] 将采样概率转化为权重ISWeights的参数
    beta_increment_per_sampling = 0.001 # beta的自增步长
    abs_err_upper = 1.  # 缩小的TD error，映射到优先级的最大值

    def __init__(self, capacity):
        self.tree = SumTree(capacity) # 实例化二叉树

    # 存储transition到二叉树中
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:]) # 获取叶节点中的最大优先级
        if max_p == 0: # 如果最大优先级为0，则设置最大优先级为1
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # 存储transition和叶节点中最大的优先级到二叉树中

    # 从二叉树中随机取n=batch_size个transition
    def sample(self, n):
        b_idx = np.empty((n,), dtype=np.int32) # 定义一个有n=batch_size个元素的一维数组，用于存储叶节点的索引
        b_memory = np.empty((n, self.tree.data[0].size)) # 定义一个有n=batch_size行，每行有data[0].size个元素的二维数组，用于存储transition
        ISWeights = np.empty((n, 1)) # 定义一个有n=batch_size行，每行有1个元素的二维数组，用于存储权重（采样概率越小，权重越大）

        pri_seg = self.tree.total_p / n       # 将根节点优先级（所有叶节点优先级之和）分为n=batch_size个段（priority segment）

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1，作为指数计算ISweights
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p # 叶节点中最小的优先级在根节点优先级的占比（选中概率），用于计算ISweights

        for i in range(n):
            a = pri_seg * i # 计算优先级段的上界
            b = pri_seg * (i + 1) # 计算优先级段的下界
            v = np.random.uniform(a, b) # 在优先级段中随机取一个值v

            idx, p, data = self.tree.get_leaf(v) # 根据值v获取叶子节点索引、优先级和数据

            b_idx[i] = idx
            b_memory[i, :] = data
            prob = p / self.tree.total_p # 叶节点优先级在根节点优先级中的占比（选中概率）
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta) # 计算ISWeights

        return b_idx, b_memory, ISWeights

    # 批量更新二叉树中的优先级
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # 自增一个小值，避免误差为0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper) # 限制误差的最大值为1
        ps = np.power(clipped_errors, self.alpha) # 将TD error转化为优先级
        for ti, p in zip(tree_idx, ps): # 批量更新优先级
            self.tree.update(ti, p)


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions, # 动作的数量
            n_features, # 特征（状态）所占数组元素的数量
            learning_rate=0.005, # 学习率
            reward_decay=0.9, # 奖励衰减率
            e_greedy=0.9, # 贪婪度
            replace_target_iter=500, # 隔多少步更新target_net的参数
            memory_size=10000, # 记忆库的容量
            batch_size=32, # 批量学习的数量
            e_greedy_increment=None, # 贪婪度的增量
            output_graph=False, # 是否输出计算图
            prioritized=True, # 是否使用优先级回放
            sess=None, # tensorflow的会话
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

        self.prioritized = prioritized    # 启用Prior_Replay_DQN算法的控制变量

        self.learn_step_counter = 0 # 学习的次数

        self._build_net() # 构建神经网络

        t_params = tf.get_collection('target_net_params') # 收集target_net的参数
        e_params = tf.get_collection('eval_net_params') # 收集eval_net的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # 用eval_net的参数更新target_net的参数

        if self.prioritized:
            self.memory = Memory(capacity=memory_size) # 定义存储transition的变量memory
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer()) # 初始化tensorflow的变量
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = [] # 记录每一次训练的cost的变化

    # 建立神经网络
    def _build_net(self):
        # 建立神经网络架构
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, trainable):
            # 建立第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
            # 建立第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,  trainable=trainable)
                out = tf.matmul(l1, w2) + b2
            return out

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # eval_net的输入
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 计算loss的输入
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights') # 计算loss的输入

        # 创建eval_net
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES] # eval_net参数的集合
            n_l1 = 20 # 第一层神经元的数量
            w_initializer = tf.random_normal_initializer(0., 0.3) # 权重的初始化器
            b_initializer = tf.constant_initializer(0.1) # 偏置的初始化器

            w = tf.Variable(w_initializer(shape=[1]), dtype=tf.float32)
            b = tf.Variable(b_initializer(shape=[1]), dtype=tf.float32)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                print(sess.run(w))
                print(sess.run(b))

            # 计算q_eval
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, True)

        # 计算loss
        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1) # 计算TD error
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval)) # 计算loss
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval)) # 计算loss
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss) # 优化器

        # 创建target_net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') # target_net的输入
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES] # target_net参数的集合
            # 计算q_next
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)


    # 存储transition到memory中
    def store_transition(self, s, a, r, s_):
        if self.prioritized:
            transition = np.hstack((s, [a, r], s_)) # 将s, a, r, s_拼接成一个数组
            self.memory.store(transition) # 存储transition到二叉树中
        else: # 如果不使用优先级回放，则按顺序存储transition到memory中
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_)) # 将s, a, r, s_拼接成一个数组
            index = self.memory_counter % self.memory_size # 计算transition在memory中的索引
            self.memory[index, :] = transition # 存储transition到memory中
            self.memory_counter += 1 # 更新memory_counter

    # 根据observation选择action
    def choose_action(self, observation):
        observation = observation[np.newaxis, :] # 将observation转化为二维矩阵
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation}) # 通过eval_net计算q_eval
            action = np.argmax(actions_value) # 选择q_eval最大的action
        else:
            action = np.random.randint(0, self.n_actions) # 随机选择action

        return action

    # 学习更新eval_net的参数
    def learn(self):
        # 间隔一定步数更新target_net的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size) # 从memory中随机取batch_size个transition（重要性采样）
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size) # 从memory中随机取batch_size个transition的索引
            batch_memory = self.memory[sample_index, :] # 根据索引取出batch_size个transition

        # 计算q_next和q_eval
        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy() # 用q_eval初始化q_target

        batch_index = np.arange(self.batch_size, dtype=np.int32) # 定义一个有batch_size个元素的一维数组，用于存储batch_size个transition的索引
        eval_act_index = batch_memory[:, self.n_features].astype(int) # 获取batch_memory中的action
        reward = batch_memory[:, self.n_features + 1] # 获取batch_memory中的reward

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) # 更新q_target

        if self.prioritized:
            # 计算TD error和cost
            _, abs_errors, cost = self.sess.run(
                [self._train_op, self.abs_errors, self.loss],
                feed_dict={self.s: batch_memory[:, :self.n_features],
                           self.q_target: q_target,
                           self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors) # 更新二叉树中的优先级
        else:
            # 计算cost
            _, cost = self.sess.run(
                [self._train_op, self.loss],
                feed_dict={self.s: batch_memory[:, :self.n_features],
                           self.q_target: q_target})

        self.cost_his.append(cost) # 记录每次学习的cost的变化

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max # 更新贪婪度
        self.learn_step_counter += 1 # 更新学习次数

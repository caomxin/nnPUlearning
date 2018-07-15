import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import tensorflow as tf
from chainer import Chain, cuda
from tensorflow.contrib import rnn


class MyClassifier(Chain):
    prior = 0

    def __call__(self, x, t, loss_func):
        self.clear()
        h = self.calculate(x)
        self.loss = loss_func(h, t)
        chainer.reporter.report({'loss': self.loss}, self)
        return self.loss

    def clear(self):
        self.loss = None

    def calculate(self, x):
        return None

    def call_reporter(self, dictionary):
        chainer.reporter.report(dictionary, self)

    def error(self, x, t):
        xp = cuda.get_array_module(x, False)
        size = len(t)
        with chainer.no_backprop_mode():
            with chainer.using_config("train", False):
                h = xp.reshape(xp.sign(self.calculate(x).data), size)
        if isinstance(h, chainer.Variable):
            h = h.data
        if isinstance(t, chainer.Variable):
            t = t.data
        result = (h != t).sum() / size
        chainer.reporter.report({'error': result}, self)
        return cuda.to_cpu(result) if xp != np else result

    def compute_prediction_summary(self, x, t):
        xp = cuda.get_array_module(x, False)
        if isinstance(t, chainer.Variable):
            t = t.data
        n_p = (t == 1).sum()
        n_n = (t == -1).sum()
        size = n_p + n_n
        with chainer.no_backprop_mode():
            with chainer.using_config("train", False):
                h = xp.reshape(xp.sign(self.calculate(x).data), size)
        if isinstance(h, chainer.Variable):
            h = h.data
        t_p = ((h == 1) * (t == 1)).sum()
        t_n = ((h == -1) * (t == -1)).sum()
        f_p = n_n - t_n
        f_n = n_p - t_p
        return t_p, t_n, f_p, f_n


class LinearClassifier(MyClassifier, Chain):
    def __init__(self, prior, dim):
        super(LinearClassifier, self).__init__(
            l=L.Linear(dim, 1)
        )
        self.prior = prior

    def calculate(self, x):
        h = self.l(x)
        return h


class DrugTargetNetwork(MyClassifier, Chain):
    def __init__(self, prior, dim):
        # self.super(DrugTargetNetwork, self).__init__()
        self.drug_input_size = 1000,
        self.timestep_size = 1,
        self.hidden_size = 256,
        self.layer_num = 2,
        self.class_num = 150,
        self.batchsize = 1,
        self.keep_prob = 0.5
        self.af = F.relu
        self.prior = prior
        self.w1 = tf.Variable(tf.truncated_normal([self.hidden_size, self.class_num], stddev=0.1), dtype=tf.float32)
        self.bias1 = tf.Variable(tf.constant(0.1, shape=[self.class_num]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.class_num], stddev=0.1), dtype=tf.float32)
        self.bias2 = tf.Variable(tf.constant(0.1, shape=[self.class_num]), dtype=tf.float32)

    def calculate(self, x):
        """
            x[0] is drug inCHI with shape([batchsize, 1, 1000])
            x[1] is target amino acid sequence with shape[batchsize, 1, 1500]
        """
        cells_drug = []
        for _ in range(self.layer_num):
            # cell
            cell = rnn.BasicLSTMCell(num_units=self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            # 添加 dropout layer, 一般只设置 output_keep_prob
            cell = rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
            cells_drug.append(cell)
        # 调用 MultiRNNCell 来实现多层 LSTM
        mlstm_cell_drug = rnn.MultiRNNCell(cells_drug, state_is_tuple=True)

        # **步骤5：用全零来初始化state
        init_state_drug = mlstm_cell_drug.zero_state(self.batch_size, dtype=tf.float32)
        outputs_drug, state_drug = tf.nn.dynamic_rnn(mlstm_cell_drug, inputs=x[0], initial_state=init_state_drug,
                                                     time_major=False,
                                                     scope="drug")
        # 最后输出
        h_state_drug = outputs_drug[:, -1, :]  # 或者 h_state = state[-1][1]

        # LSTM1 预测结果
        y1_pre = tf.nn.softmax(tf.matmul(h_state_drug, self.w1) + self.bias1)

        # LSTM2 for target
        cells_target = []
        for _ in range(self.layer_num):
            cell_target = rnn.BasicLSTMCell(num_units=self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            cell_target = rnn.DropoutWrapper(cell=cell_target, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
            cells_target.append(cell_target)
        mlstm_cell_target = rnn.MultiRNNCell(cells_target, state_is_tuple=True)

        init_state_target = mlstm_cell_target.zero_state(self.batch_size, dtype=tf.float32)

        outputs_target, state_target = tf.nn.dynamic_rnn(mlstm_cell_target, inputs=x[1],
                                                         initial_state=init_state_target, time_major=False,
                                                         scope="target")

        h_state_target = outputs_target[:, -1, :]  # 或者 h_state = state[-1][1]

        # LSTM2 预测结果
        y2_pre = tf.nn.softmax(tf.matmul(h_state_target, self.w2) + self.bias2)

        relationship_calculated = tf.matmul(y1_pre, tf.transpose(y2_pre))

        return relationship_calculated


class ThreeLayerPerceptron(MyClassifier, Chain):
    def __init__(self, prior, dim):
        super(ThreeLayerPerceptron, self).__init__(l1=L.Linear(dim, 100),
                                                   l2=L.Linear(100, 1))
        self.af = F.relu
        self.prior = prior

    def calculate(self, x):
        h = self.l1(x)
        h = self.af(h)
        h = self.l2(h)
        return h


class MultiLayerPerceptron(MyClassifier, Chain):
    def __init__(self, prior, dim):
        super(MultiLayerPerceptron, self).__init__(l1=L.Linear(dim, 300, nobias=True),
                                                   b1=L.BatchNormalization(300),
                                                   l2=L.Linear(300, 300, nobias=True),
                                                   b2=L.BatchNormalization(300),
                                                   l3=L.Linear(300, 300, nobias=True),
                                                   b3=L.BatchNormalization(300),
                                                   l4=L.Linear(300, 300, nobias=True),
                                                   b4=L.BatchNormalization(300),
                                                   l5=L.Linear(300, 1))
        self.af = F.relu
        self.prior = prior

    def calculate(self, x):
        h = self.l1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.l2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.l3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.l4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.l5(h)
        return h


class CNN(MyClassifier, Chain):
    def __init__(self, prior, dim):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 96, 3, pad=1),
            conv2=L.Convolution2D(96, 96, 3, pad=1),
            conv3=L.Convolution2D(96, 96, 3, pad=1, stride=2),
            conv4=L.Convolution2D(96, 192, 3, pad=1),
            conv5=L.Convolution2D(192, 192, 3, pad=1),
            conv6=L.Convolution2D(192, 192, 3, pad=1, stride=2),
            conv7=L.Convolution2D(192, 192, 3, pad=1),
            conv8=L.Convolution2D(192, 192, 1),
            conv9=L.Convolution2D(192, 10, 1),
            b1=L.BatchNormalization(96),
            b2=L.BatchNormalization(96),
            b3=L.BatchNormalization(96),
            b4=L.BatchNormalization(192),
            b5=L.BatchNormalization(192),
            b6=L.BatchNormalization(192),
            b7=L.BatchNormalization(192),
            b8=L.BatchNormalization(192),
            b9=L.BatchNormalization(10),
            fc1=L.Linear(None, 1000),
            fc2=L.Linear(1000, 1000),
            fc3=L.Linear(1000, 1),
        )
        self.af = F.relu
        self.prior = prior

    def calculate(self, x):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h
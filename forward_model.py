from collections import OrderedDict
import ops
import tensorflow as tf
import common


class ForwardModel(object):
    def __init__(self, state_size, action_size, encoding_size, lr, ensemble):
        self.state_size = state_size
        self.action_size = action_size
        self.encoding_size = encoding_size

        self.lr = lr
        self.num = ensemble

    def forward(self,
                inputs,
                labels=None,
                is_training=True,
                dtype=tf.float32,
                w_dict=None,
                ex_wts=None,
                reuse=None):
        """Builds a simple LeNet.
        
        :param inputs:            [Tensor]    Inputs.
        :param labels:            [Tensor]    Labels.
        :param is_training:       [bool]      Whether in training mode, default True.
        :param dtype:             [dtype]     Data type, default tf.float32.
        :param w_dict:            [dict]      Dictionary of weights, default None.
        :param ex_wts:            [Tensor]    Example weights placeholder, default None.
        :param reuse:             [bool]      Whether to reuse variables, default None.
        """

        if w_dict is None:
            w_dict = {}

        def _get_var(name, shape, dtype, initializer):
            key = tf.get_variable_scope().name + '/' + name
            if key in w_dict:
                return w_dict[key]
            else:
                var = tf.get_variable(name, shape, dtype, initializer=initializer)
                w_dict[key] = var
                return var

        def _repeat_var(var):
            # print(var.shape)
            repeated_var = tf.tile(var, [self.num, 1])
            dim = repeated_var.shape[-1]
            repeated_var = tf.reshape(repeated_var, [self.num, -1, dim])
            # print('repeated_var.shape', repeated_var.shape)
            return repeated_var

        with tf.variable_scope('Model', reuse=reuse):
            state_raw = tf.cast(inputs[0], dtype)
            action_raw = tf.cast(inputs[1], dtype)
            gru_state_raw = tf.cast(inputs[2], dtype)

            state = _repeat_var(state_raw)
            action = _repeat_var(action_raw)
            gru_state = _repeat_var(gru_state_raw)

            if is_training:
                labels_raw = tf.cast(labels, dtype)
                labels = _repeat_var(labels_raw)

            w_init = tf.random_normal_initializer(stddev=0.15)
            w1 = _get_var('w1', [self.num, self.state_size, self.encoding_size], dtype, initializer=w_init)
            w2 = _get_var('w2', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            w3 = _get_var('w3', [self.num, self.action_size, self.encoding_size], dtype, initializer=w_init)
            w4 = _get_var('w4', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            w5 = _get_var('w5', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            w6 = _get_var('w6', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            w7 = _get_var('w7', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            w8 = _get_var('w8', [self.num, self.encoding_size, self.state_size], dtype, initializer=w_init)

            b1 = _get_var('b1', [self.num, 1, self.encoding_size], dtype, initializer=w_init)
            b2 = _get_var('b2', [self.num, 1, self.encoding_size], dtype, initializer=w_init)
            b3 = _get_var('b3', [self.num, 1, self.encoding_size], dtype, initializer=w_init)
            b4 = _get_var('b4', [self.num, 1, self.encoding_size], dtype, initializer=w_init)
            b5 = _get_var('b5', [self.num, 1, self.encoding_size], dtype, initializer=w_init)
            b6 = _get_var('b6', [self.num, 1, self.encoding_size], dtype, initializer=w_init)
            b7 = _get_var('b7', [self.num, 1, self.encoding_size], dtype, initializer=w_init)
            b8 = _get_var('b8', [self.num, 1, self.state_size], dtype, initializer=w_init)

            Wxr = _get_var('weights_xr', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            Wxz = _get_var('weights_xz', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            Wxh = _get_var('weights_xh', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            Whr = _get_var('weights_hr', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            Whz = _get_var('weights_hz', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            Whh = _get_var('weights_hh', [self.num, self.encoding_size, self.encoding_size], dtype, initializer=w_init)
            br = _get_var('biases_r', [self.num, 1, self.encoding_size], dtype, initializer=w_init)
            bz = _get_var('biases_z', [self.num, 1, self.encoding_size], dtype, initializer=w_init)
            bh = _get_var('biases_h', [self.num, 1, self.encoding_size], dtype, initializer=w_init)

            # State embedding
            # state_embedder1 = ops.dense(state, self.state_size, self.encoding_size, tf.nn.relu, "encoder1_state", reuse)
            state_embedder1 = tf.nn.relu(tf.add(tf.matmul(state, w1), b1, name='encoder1_state'))

            # gru_state = ops.gru(state_embedder1, gru_state, self.encoding_size, self.encoding_size, 'gru1', reuse)
            x, h_ = state_embedder1, gru_state
            r = tf.sigmoid(tf.matmul(x, Wxr) + tf.matmul(h_, Whr) + br)
            z = tf.sigmoid(tf.matmul(x, Wxz) + tf.matmul(h_, Whz) + bz)
            h_hat = tf.tanh(tf.matmul(x, Wxh) + tf.matmul(tf.multiply(r, h_), Whh) + bh)
            gru_state = tf.multiply((1 - z), h_hat) + tf.multiply(z, h_)

            # state_embedder2 = ops.dense(gru_state, self.encoding_size, self.encoding_size, tf.sigmoid, "encoder2_state", reuse)
            state_embedder2 = tf.sigmoid(tf.add(tf.matmul(gru_state, w2), b2, name='encoder2_state'))


            # Action embedding
            # action_embedder1 = ops.dense(action, self.action_size, self.encoding_size, tf.nn.relu, "encoder1_action", reuse)
            action_embedder1 = tf.nn.relu(tf.add(tf.matmul(action, w3), b3, name='encoder1_action'))

            # action_embedder2 = ops.dense(action_embedder1, self.encoding_size, self.encoding_size, tf.sigmoid, "encoder2_action", reuse)
            action_embedder2 = tf.sigmoid(tf.add(tf.matmul(action_embedder1, w4), b4, name='encoder2_action'))


            # Joint embedding
            joint_embedding = tf.multiply(state_embedder2, action_embedder2)


            # Next state prediction
            # hidden1 = ops.dense(joint_embedding, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder3", reuse)
            hidden1 = tf.nn.relu(tf.add(tf.matmul(joint_embedding, w5), b5, name='encoder3'))

            # hidden2 = ops.dense(hidden1, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder4", reuse)
            hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, w6), b6, name='encoder4'))

            # hidden3 = ops.dense(hidden2, self.encoding_size, self.encoding_size, tf.nn.relu, "decoder1", reuse)
            hidden3 = tf.nn.relu(tf.add(tf.matmul(hidden2, w7), b7, name='decoder1'))

            # next_state = ops.dense(hidden3, self.encoding_size, self.state_size, None, "decoder2", reuse)
            next_state = tf.add(tf.matmul(hidden3, w8), b8, name='decoder2')

            gru_state = tf.cast(gru_state, tf.float64)

            if not is_training:
                next_state = tf.reduce_mean(next_state, axis=0)
                gru_state = tf.reduce_mean(gru_state, axis=0)
                return next_state, gru_state

            if ex_wts is None:
                # Average loss.
                loss = tf.reduce_mean(tf.square(labels - next_state))
            else:
                # Weighted loss.
                loss = tf.reduce_mean(tf.square(labels - next_state), axis=-1)
                loss = tf.reduce_mean(loss * ex_wts)
        return w_dict, loss

    def reweight_autodiff(self,
                          inp_a,
                          label_a,
                          inp_b,
                          label_b,
                          bsize_a,
                          bsize_b,
                          eps=0.0,
                          gate_gradients=1):
        """Reweight examples using automatic differentiation.
        :param inp_a:             [Tensor]    Inputs for the noisy pass.
        :param label_a:           [Tensor]    Labels for the noisy pass.
        :param inp_b:             [Tensor]    Inputs for the clean pass.
        :param label_b:           [Tensor]    Labels for the clean pass.
        :param bsize_a:           [int]       Batch size for the noisy pass.
        :param bsize_b:           [int]       Batch size for the clean pass.
        :param eps:               [float]     Minimum example weights, default 0.0.
        :param gate_gradients:    [int]       Tensorflow gate gradients, reduce concurrency.
        """
        ex_wts_a = tf.zeros([self.num, bsize_a], dtype=tf.float32)
        ex_wts_b = tf.ones([self.num, bsize_b], dtype=tf.float32) / float(bsize_b)
        w_dict, loss_a = self.forward(
            inp_a, label_a, ex_wts=ex_wts_a, is_training=True, reuse=True)
        var_names = w_dict.keys()
        var_list = [w_dict[kk] for kk in var_names]
        grads = tf.gradients(loss_a, var_list, gate_gradients=gate_gradients)

        var_list_new = [vv - gg for gg, vv in zip(grads, var_list)]
        w_dict_new = dict(zip(var_names, var_list_new))
        _, loss_b = self.forward(
            inp_b, label_b, ex_wts=ex_wts_b, is_training=True, reuse=True, w_dict=w_dict_new)
        grads_ex_wts = tf.gradients(loss_b, [ex_wts_a], gate_gradients=gate_gradients)[0]
        ex_weight = -grads_ex_wts
        ex_weight_plus = tf.maximum(ex_weight, eps)
        ex_weight_sum = tf.reduce_sum(ex_weight_plus)
        ex_weight_sum += tf.to_float(tf.equal(ex_weight_sum, 0.0))
        ex_weight_norm = ex_weight_plus / ex_weight_sum
        return ex_weight_norm

    def train(self, x_, y_, ex_wts):
        with tf.name_scope('Train'):
            _, self.loss_c = self.forward(
                x_, y_, is_training=True, dtype=tf.float32, w_dict=None, ex_wts=ex_wts, reuse=None)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_c)

    def reweight(self, x_, y_, x_val_, y_val_, bsize_a, bsize_b):
        self.ex_weights_ = self.reweight_autodiff(x_, y_, x_val_, y_val_, bsize_a, bsize_b, eps=0.0, gate_gradients=1)

    def predict(self, x_, y_):
        with tf.name_scope('Predict'):
            self.prediction = self.forward(
                x_, y_, is_training=False, dtype=tf.float32, w_dict=None, ex_wts=None, reuse=True)

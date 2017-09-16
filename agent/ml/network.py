import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def _make_network(inpt, rotate_inpt, movement_inpt, rnn_state_tuple, num_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        conv_out = layers.fully_connected(out, 256, activation_fn=tf.nn.relu)

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            rnn_in = tf.expand_dims(conv_out, [0])
            step_size = tf.shape(inpt)[:1]
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, initial_state=rnn_state_tuple,
                    sequence_length=step_size, time_major=False)
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        encode = layers.fully_connected(rnn_out, 128, activation_fn=None,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='encode')

        value = layers.fully_connected(rnn_out, 1, activation_fn=None,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='value')

        rotate_inpt = tf.expand_dims(rotate_inpt, 1)
        movement_inpt = tf.expand_dims(movement_inpt, 1)

        out = tf.concat([rnn_out, rotate_inpt, movement_inpt], 1)
        out = layers.fully_connected(out, 128, activation_fn=tf.nn.relu,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='concated_layer')

        hidden_place_cell = layers.fully_connected(out, 32, activation_fn=tf.nn.relu,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='hidden_place_cell')
        place_cell = layers.fully_connected(hidden_place_cell, 1, activation_fn=None,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='place_cell')

        hidden_head_cell = layers.fully_connected(out, 32, activation_fn=tf.nn.relu,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='hidden_head_cell')
        head_cell = layers.fully_connected(hidden_head_cell, 1, activation_fn=None,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='head_cell')

        hidden_grid_cell = layers.fully_connected(out, 32, activation_fn=tf.nn.relu,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='hidden_grid_cell')
        grid_cell = layers.fully_connected(hidden_grid_cell, 1, activation_fn=None,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='grid_cell')

        concated_cells = tf.concat([hidden_place_cell, hidden_head_cell, hidden_grid_cell], 1)
        ca1 = layers.fully_connected(concated_cells, 32, activation_fn=tf.nn.relu,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None, scope='ca1')

    return encode, value, lstm_state, place_cell, head_cell, grid_cell, ca1

def make_network():
    return lambda *args, **kwargs: _make_network(*args, **kwargs)

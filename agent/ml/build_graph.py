import numpy as np
import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(model, dnds, num_actions, optimizer, scope='a3c', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_input = tf.placeholder(tf.float32, [None, 10240], name='obs')
        rnn_state_ph0 = tf.placeholder(tf.float32, [1, 256], name='rnn_state0')
        rnn_state_ph1 = tf.placeholder(tf.float32, [1, 256], name='rnn_state1')
        rotate_input = tf.placeholder(tf.float32, [None], name='rotation')
        movement_input = tf.placeholder(tf.float32, [None], name='movement')

        actions_ph = tf.placeholder(tf.uint8, [None], name='action')
        target_values_ph = tf.placeholder(tf.float32, [None], name='value')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(rnn_state_ph0, rnn_state_ph1)
        place_ph = tf.placeholder(tf.float32, [None], name='place')
        head_ph = tf.placeholder(tf.float32, [None], name='head')
        grid_ph = tf.placeholder(tf.float32, [None], name='grid')

        encode, value, state_out, place_cell, head_cell, grid_cell, ca1 = model(
                obs_input, rotate_input, movement_input, rnn_state_tuple, num_actions, scope='model')

        with tf.name_scope('dnd'):
            concated_encode = tf.concat([encode, ca1], 1)
            probs = []
            for i, dnd in enumerate(dnds):
                keys, values = tf.py_func(dnd.lookup, [concated_encode], [tf.float32, tf.float32])
                square_diff = tf.square(keys - tf.expand_dims(concated_encode, 1))
                distances = tf.reduce_sum(square_diff, axis=2) + 1e-3
                weights = 1 / distances
                normalized_weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
                probs.append(tf.reduce_sum(normalized_weights * values, axis=1))
            policy = tf.nn.softmax(tf.transpose(probs))

            actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
            responsible_outputs = tf.reduce_sum(policy * actions_one_hot, [1])

        with tf.name_scope('loss'):
            log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
            value_loss = tf.nn.l2_loss(target_values_ph - tf.reshape(value, [-1]), name='value_loss')

            entropy = -tf.reduce_sum(policy * log_policy)
            policy_loss = -tf.reduce_sum(tf.reduce_sum(
                    tf.multiply(log_policy, actions_one_hot)) * advantages_ph + entropy * 0.01, name='policy_loss')

            place_loss = tf.nn.l2_loss(place_ph - place_cell, name='place_loss')
            head_loss = tf.nn.l2_loss(head_ph - head_cell, name='head_loss')
            grid_loss = tf.nn.l2_loss(grid_ph - grid_cell, name='grid_loss')

            loss = 0.5 * value_loss + policy_loss + 0.1 * place_loss + 0.1 * head_loss + 0.1 * grid_loss
            loss_summary = tf.summary.scalar('{}_loss'.format(scope), loss)

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, local_vars), 40.0)

        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        optimize_expr = optimizer.apply_gradients(zip(gradients, global_vars))

        update_local_expr = []
        for local_var, global_var in zip(local_vars, global_vars):
            update_local_expr.append(local_var.assign(global_var))
        update_local_expr = tf.group(*update_local_expr)
        update_local = util.function([], [], updates=[update_local_expr])

        train = util.function(
            inputs=[
                obs_input, rnn_state_ph0, rnn_state_ph1, rotate_input, movement_input,
                        actions_ph, target_values_ph, advantages_ph, place_ph, head_ph, grid_ph
            ],
            outputs=[loss_summary, loss],
            updates=[optimize_expr]
        )

        action_dist = util.function([obs_input, rnn_state_ph0, rnn_state_ph1, rotate_input, movement_input], policy)

        state_value = util.function([obs_input, rnn_state_ph0, rnn_state_ph1], value)

        act = util.function(inputs=[obs_input, rnn_state_ph0, rnn_state_ph1,
                rotate_input, movement_input], outputs=[policy, state_out, concated_encode])

    return act, train, update_local, action_dist, state_value

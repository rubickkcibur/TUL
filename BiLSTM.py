import numpy as np
import os
import gensim
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

def RNN(x, weights, biases, keep_prob, hidden_size,seq_length):
    x = tf.transpose(x, [1, 0, 2])

    fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0,
                                                state_is_tuple=True)
    fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)
    bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0,
                                                state_is_tuple=True)
    bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)

    cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)

    (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32,
                                                        time_major=True,sequence_length=seq_length)
    fw_output, fb_output = outputs

    new_outputs = tf.concat(outputs,2)

    val = tf.matmul(new_outputs[-1], weights['out']) + biases['out']

    return val
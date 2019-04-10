# -*- coding: utf-8 -*-

# Copyright 2018 Leiming Du Authors. All Rights Reserved.
#
# Licensed under the "Anti 996" License, Version 1.0;
#
# "Anti 996" License Version 1.0 (Draft)
# Permission is hereby granted to any individual or legal entity
# obtaining a copy of this licensed work (including the source code,
# documentation and/or related items, hereinafter collectively referred
# to as the "licensed work"), free of charge, to deal with the licensed
# work for any purpose, including without limitation, the rights to use,
# reproduce, modify, prepare derivative works of, distribute, publish 
# and sublicense the licensed work, subject to the following conditions:
#
# 1. The individual or the legal entity must conspicuously display,
# without modification, this License and the notice on each redistributed 
# or derivative copy of the Licensed Work.
#
# 2. The individual or the legal entity must strictly comply with all
# applicable laws, regulations, rules and standards of the jurisdiction
# relating to labor and employment where the individual is physically
# located or where the individual was born or naturalized; or where the
# legal entity is registered or is operating (whichever is stricter). In
# case that the jurisdiction has no such laws, regulations, rules and
# standards or its laws, regulations, rules and standards are
# unenforceable, the individual or the legal entity are required to
# comply with Core International Labor Standards.
#
# 3. The individual or the legal entity shall not induce, metaphor or force
# its employee(s), whether full-time or part-time, or its independent
# contractor(s), in any methods, to agree in oral or written form, to
# directly or indirectly restrict, weaken or relinquish his or her
# rights or remedies under such laws, regulations, rules and standards
# relating to labor and employment as mentioned above, no matter whether
# such written or oral agreement are enforceable under the laws of the
# said jurisdiction, nor shall such individual or the legal entity
# limit, in any methods, the rights of its employee(s) or independent
# contractor(s) from reporting or complaining to the copyright holder or
# relevant authorities monitoring the compliance of the license about
# its violation(s) of the said license.
#
# THE LICENSED WORK IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN ANY WAY CONNECTION WITH THE
# LICENSED WORK OR THE USE OR OTHER DEALINGS IN THE LICENSED WORK.
# ==============================================================================
# @Time    : 3/12/18 17:26
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: keras_model.py
import sys
sys.path.append('../')

from params_and_config import paramsAndConfig
from Preproessing.dataUtils import data_utils
import tensorflow as tf
from crf_layer import crf_layer
import tensorflow_hub as hub
import numpy as np
import math
import pandas as pd

class KerasSaSentTensorflow(tf.keras.Model):
    def __init__(self):
        super(KerasSaSentTensorflow, self).__init__()

        # inner tool and config file
        self.data_tool = data_utils()
        self.param_and_config = paramsAndConfig()
        self.params = self.param_and_config.params
        self.config = self.param_and_config.config

        self.dic = self.data_tool.read_gzip_serialized_file(self.config.get('dic_path'))
        self.word2id = {k: v for v, k in enumerate(self.dic)}

        if self.params.get('word_embedding'):
            # Here, when reading, pd is faster than np, np is smaller than serialized.
            # So the best solution is to save with np and read with pd.

            # self.embedding = self.data_tool.read_gzip_serialized_file(self.config.get('embedding_path'))
            # self.embedding = np.loadtxt(fname=self.config.get('embedding_path'), dtype=np.float32)

            self.embedding = pd.read_csv(filepath_or_buffer=self.config.get('embedding_path'), sep=' ').values
            self.embedding = tf.Variable(
                initial_value=self.embedding, trainable=False, name='word_embedding_table', dtype=tf.float32
            )
        self.transition = tf.get_variable(name='transition', shape=[2, 2], dtype=tf.float32, trainable=True,
                                          initializer=tf.random_normal_initializer)

        # Build some parameters of the model.

        # mask embedding part
        tf.logging.info('Warning: randomly initializing mask(binary) embeddings!')
        self._mask_embedding = tf.get_variable(
            name='mask_embedding',
            dtype=tf.float32,
            shape=[2, self.params.get('mask_dim')],
            trainable=True,
            initializer=tf.uniform_unit_scaling_initializer()
        )

    def call(self, inputs, training=True):
        """
        Given inputs, return the logits.
        :param features:
        :param training:
        :return:
        """
        inputs_seq, masks, length = inputs
        length = tf.squeeze(length)
        # Given inputs, to generate the embedding as the input to next layer.
        with tf.variable_scope(name_or_scope='input_embedding_scope', reuse=tf.AUTO_REUSE) as in_em_scope:
            # use elmo as the input embedding
            if self.params.get('elmo'):
                elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
                # change inputs to a list of words to fit into the elmo
                for i in range(len(inputs)):
                    inputs_seq[i] = [self.dic[v] for v in inputs_seq[i]]

                # Size of input_embedding: batch_size * max_length * 1024(default)
                input_embedding = elmo(inputs={
                    'tokens': inputs_seq,
                    'sequence_len': length
                }, signature='tokens', as_dict=True)['elmo']

            # use Bert as the input embedding
            if self.params.get('bert'):
                # TODO embed bert model here.
                pass

            # Use Glove/word2vec embedding as the input
            if self.params.get('word_embedding'):
                assert self.embedding is not None
                input_embedding = tf.nn.embedding_lookup(self.embedding, inputs_seq, name='input_embedding')
            # Use char embedding as the supplementary embedding
            if self.params.get('char_embedding'):
                # TODO embed char embedding here, need to think about how to store the instance.
                pass

            mask_embedding = tf.nn.embedding_lookup(self._mask_embedding, masks, name='mask_embedding')

            # concat input and mask embedding
            input_embedding = tf.concat([input_embedding, mask_embedding], axis=-1)

        with tf.variable_scope('lstm_part', reuse=tf.AUTO_REUSE) as lstm_part:
            lstm_output = input_embedding
            for i in range(self.params.get('layer_num')):
                lstm_output = self.add_lstm_layer(inputs=lstm_output, length=length, layer_name=i)

            if self.params.get('if_residual'):
                lstm_output = input_embedding + tf.layers.dense(inputs=lstm_output,
                                                                units=self.params.get('word_dimension') + self.params.get(
                                                                    'mask_dim'))

        # CRF layer
        with tf.variable_scope('crf_layer', reuse=tf.AUTO_REUSE) as crf_layer_layer:
            crf_input = tf.layers.dense(lstm_output, units=2, bias_initializer=tf.glorot_uniform_initializer())
            crf_layer_ = crf_layer(inputs=crf_input, sequence_lengths=length, transition_prob=self.transition)
            crf_output = crf_layer_.crf_output_prob()[:, :, -1]  # The size should be batch_size * seq_len

            # expand crf_output's shape to batch_size * 1 * seq_len for batch matrix multipcation
            crf_output = tf.expand_dims(crf_output, axis=1)

            # also can chose matmul
            # sentiment_vector = tf.squeeze(
            #     tf.einsum('aij,ajk->aik', crf_output, lstm_output))  # output shape is batch_size * embedding_dim

            # sentiment_vector = tf.squeeze(tf.matmul(crf_output, lstm_output))
            sentiment_vector = tf.matmul(crf_output, lstm_output)

        # logits layer
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE) as logits_layer:
            logits = tf.layers.dense(inputs=sentiment_vector, units=self.params.get('n_classes'), activation='softmax',
                                     bias_initializer=tf.glorot_uniform_initializer())

            return logits

    def add_lstm_layer(self, inputs, length, layer_name):
        """
        Add one bidirectional layer.
        :param intputs: inputs.
        :param length: a batch of length.
        :return: concated lstm output.
        """
        with tf.variable_scope(str(layer_name) + '_layer'):
            cell_fw = tf.contrib.rnn.LSTMCell(self.params.get('hidden_dimension'))
            cell_bw = tf.contrib.rnn.LSTMCell(self.params.get('hidden_dimension'))
            if self.params.get('layer_norm'):
                cell_fw = tf.contrib.layers.layer_norm(cell_fw)
                cell_bw = tf.contrib.layers.layer_norm(cell_bw)
            elif self.params.get('if_dropout'):
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                        output_keep_prob=1 - self.params.get('dropout_rate'))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                        output_keep_prob=1 - self.params.get('dropout_rate'))
            (lstm_output_fw, lstm_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs=inputs, sequence_length=length, dtype=tf.float32
            )
            return tf.concat([lstm_output_fw, lstm_output_bw], axis=-1)
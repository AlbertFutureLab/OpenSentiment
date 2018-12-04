# -*- coding: utf-8 -*-

# Copyright 2018 Leiming Du Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
            # self.embedding = self.data_tool.read_gzip_serialized_file(self.config.get('embedding_path'))
            self.embedding = np.loadtxt(fname=self.config.get('embedding_path'), dtype=np.float32)
        self.transition = tf.get_variable(name='transition', shape=[2, 2], dtype=tf.float32, trainable=True)

        # Build some parameters of the model.

        # mask embedding part
        tf.logging.info('Warning: randomly initializing mask(binary) embeddings!')
        self._mask_embedding = tf.get_variable(
            name='mask_embedding',
            dtype=tf.float32,
            shape=[2, self.params.get('mask_dim')],
            trainable=True
        )

    def call(self, inputs, training=True):
        """
        Given inputs, return the logits.
        :param features:
        :param training:
        :return:
        """
        inputs, masks, length = inputs
        length = tf.squeeze(length)
        # Given inputs, to generate the embedding as the input to next layer.
        with tf.variable_scope(name_or_scope='input_embedding_scope', reuse=tf.AUTO_REUSE) as in_em_scope:
            # use elmo as the input embedding
            if self.params.get('elmo'):
                elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
                # change inputs to a list of words to fit into the elmo
                for i in range(len(inputs)):
                    inputs[i] = [self.dic[v] for v in inputs[i]]

                # Size of input_embedding: batch_size * max_length * 1024(default)
                input_embedding = elmo(inputs={
                    'tokens': inputs,
                    'sequence_len': length
                }, signature='tokens', as_dict=True)['elmo']

            # use Bert as the input embedding
            if self.params.get('bert'):
                # TODO embed bert model here.
                pass

            # Use Glove/word2vec embedding as the input
            if self.params.get('word_embedding'):
                assert self.embedding is not None
                input_embedding = tf.nn.embedding_lookup(self.embedding, inputs, name='input_embedding')
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

            lstm_output = input_embedding + tf.layers.dense(inputs=lstm_output,
                                                            units=self.params.get('word_dimension') + self.params.get(
                                                                'mask_dim'))

        # CRF layer
        with tf.variable_scope('crf_layer', reuse=tf.AUTO_REUSE) as crf_layer_layer:
            crf_input = tf.layers.dense(lstm_output, units=2)
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
            logits = tf.layers.dense(inputs=sentiment_vector, units=self.params.get('n_classes'), activation='softmax')

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
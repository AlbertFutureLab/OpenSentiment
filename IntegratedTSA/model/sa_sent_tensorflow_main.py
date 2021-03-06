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
# @Time    : 1/12/18 09:31
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: sa_sent_tensorflow_main.py
import sys, os
import numpy as np

# To add current dir to search path to prevent some errors
sys.path.append('../')

from params_and_config import paramsAndConfig
from Preproessing.dataUtils import data_utils
import tensorflow as tf
from crf_layer import crf_layer
import tensorflow_hub as hub
import pandas as pd
import math

class Sa_Sent_Tensorflow(object):
    def __init__(self):
        # inner tool and config file
        self.data_tool = data_utils()
        self.param_and_config = paramsAndConfig()
        self.params = self.param_and_config.params
        self.config = self.param_and_config.config

        self.dic = self.data_tool.read_gzip_serialized_file(self.config.get('dic_path'))
        self.word2id = {k:v for v, k in enumerate(self.dic)}

        if self.params.get('word_embedding'):
            self.embedding = pd.read_csv(filepath_or_buffer=self.config.get('embedding_path'), sep=' ').values
            self.embedding = tf.Variable(
                initial_value=self.embedding, trainable=False, name='word_embedding_table', dtype=tf.float32
            )

        self.transition = tf.get_variable(name='transition', shape=[2, 2], dtype=tf.float32, trainable=True,
                                          initializer=tf.random_normal_initializer)

    def my_model(self, features, labels, mode):
        """
        The SA-Sent-tensorflow model, mainly three BiLSTMs + CRF.
        :param features: [inputs, masks, length]
        :param labels: a batch of labels.
        :param mode: estimator mode.
        :return: tf.estimator.EstimatorSpec object
        """
        inputs, masks, length = features
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

            # mask embedding part
            tf.logging.info('Warning: randomly initializing mask(binary) embeddings!')
            _mask_embedding = tf.get_variable(
                name='_mask_embedding',
                dtype=tf.float32,
                shape=[2, self.params.get('mask_dim')],
                trainable=True,
                initializer=tf.uniform_unit_scaling_initializer()
            )
            mask_embedding = tf.nn.embedding_lookup(_mask_embedding, masks, name='mask_embedding')

            # concat input and mask embedding
            input_embedding = tf.concat([input_embedding, mask_embedding], axis=-1)

        with tf.variable_scope('lstm_part', reuse=tf.AUTO_REUSE) as lstm_part:
            lstm_output = input_embedding
            for i in range(self.params.get('layer_num')):
                lstm_output = self.add_lstm_layer(inputs=lstm_output, length=length, layer_name=i)

            if self.params.get('if_residual'):
                lstm_output = input_embedding + tf.layers.dense(inputs=lstm_output,
                                                            units=self.params.get('word_dimension'),
                                                                bias_initializer=tf.glorot_uniform_initializer())
        # CRF layer
        with tf.variable_scope('crf_layer',reuse=tf.AUTO_REUSE) as crf_layer_layer:
            crf_input = tf.layers.dense(lstm_output, units=2, bias_initializer=tf.glorot_uniform_initializer())
            crf_layer_ = crf_layer(inputs=crf_input, sequence_lengths=length, transition_prob=self.transition)
            crf_output = crf_layer_.crf_output_prob()[:, :, -1]  # The size should be batch_size * seq_len

            # expand crf_output's shape to batch_size * 1 * seq_len for batch matrix multipcation
            crf_output = tf.expand_dims(crf_output, axis=1)

            # There are some problems. using tf.einsum, the tensor would not get a defined shape. Then where would be some
            # problems in later usage.
            # sentiment_vector = tf.squeeze(
            #     tf.einsum('aij,ajk->aik', crf_output, lstm_output))  # output shape is batch_size * embedding_dim

            sentiment_vector = tf.squeeze(tf.matmul(crf_output, lstm_output))

        # logits layer
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE) as logits_layer:
            logits = tf.layers.dense(sentiment_vector, self.params.get('n_classes'), bias_initializer=tf.glorot_uniform_initializer())

            # Compute predictions
            predicted_classes = tf.argmax(logits, axis=-1)
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'sentiment': predicted_classes,
                    'prob': tf.nn.softmax(logits),
                    'logits': logits
                }
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # Compute loss.
            losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(indices=labels, depth=3), logits=logits))
            tf.summary.scalar('loss', losses)

            # Compute evaluation metrics.
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=predicted_classes,
                                           name='accuracy_op')[1]
            metrics = {'accuracy': accuracy}
            # Write to tensorboard
            tf.summary.scalar('accuracy', accuracy[1])

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=losses, eval_metric_ops=metrics
                )

            # Create the train_op in training mode
            assert mode == tf.estimator.ModeKeys.TRAIN
            if self.config.get('learning_algorithm') == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.get('learning_rate'))
            elif self.config.get('learning_algorithm') == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.config.get('learning_rate'))
            else:
                raise Exception('You must specify the learning algorithm.')
            train_op = optimizer.minimize(loss=losses, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=losses, train_op=train_op
            )

    def model_only(self, inputs, masks, length, labels, mode='train'):
        """
        Given inputs, ..., return the accuracy or logits depending on the mode.
        :param inputs: a batch of list of ids.
        :param masks: a batch of list of ids.
        :param length: a batch of ints.
        :param labels: a batch of labels.
        :param mode: ['train', 'val', 'predict'], 'train' default.
        :return: depending on the mode.
        """
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

            # mask embedding part
            tf.logging.info('Warning: randomly initializing mask(binary) embeddings!')
            _mask_embedding = tf.get_variable(
                name='_mask_embedding',
                dtype=tf.float32,
                shape=[2, self.params.get('mask_dim')],
                trainable=True,
                initializer=tf.uniform_unit_scaling_initializer()
            )
            mask_embedding = tf.nn.embedding_lookup(_mask_embedding, masks, name='mask_embedding')

            # concat input and mask embedding
            input_embedding = tf.concat([input_embedding, mask_embedding], axis=-1)

        with tf.variable_scope('lstm_part', reuse=tf.AUTO_REUSE) as lstm_part:
            lstm_output = input_embedding
            for i in range(self.params.get('layer_num')):
                lstm_output = self.add_lstm_layer(inputs=lstm_output, length=length, layer_name=i)

            if self.params.get('if_residual'):
                lstm_output = input_embedding + tf.layers.dense(inputs=lstm_output,
                                                            units=self.params.get('word_dimension')+self.params.get('mask_dim'),
                                                                bias_initializer=tf.glorot_normal_initializer())

        # CRF layer
        with tf.variable_scope('crf_layer', reuse=tf.AUTO_REUSE) as crf_layer_layer:
            crf_input = tf.layers.dense(lstm_output, units=2)
            crf_layer_ = crf_layer(inputs=crf_input, sequence_lengths=length, transition_prob=self.transition)
            crf_output = crf_layer_.crf_output_prob()[:, :, -1]  # The size should be batch_size * seq_len

            # expand crf_output's shape to batch_size * 1 * seq_len for batch matrix multipcation
            crf_output = tf.expand_dims(crf_output, axis=1)

            # There are some problems. using tf.einsum, the tensor would not get a defined shape. Then where would be some
            # problems in later usage.
            # sentiment_vector = tf.einsum('aij,ajk->aik', crf_output, lstm_output)  # output shape is batch_size * embedding_dim

            # sentiment_vector = tf.squeeze(tf.matmul(crf_output, lstm_output))
            sentiment_vector = tf.matmul(crf_output, lstm_output)

        # logits layer
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE) as logits_layer:
            # logits = tf.layers.dense(inputs=sentiment_vector, units=self.params.get('n_classes'))
            logits = tf.layers.dense(inputs=sentiment_vector, units=3)

            logits = tf.squeeze(logits)
            labels = tf.squeeze(labels)

            # Compute predictions
            predicted_classes = tf.argmax(logits, axis=-1)
            predictions = {
                    'sentiment': predicted_classes,
                    'prob': tf.nn.softmax(logits),
                    'logits': logits
                }

            # Compute loss.
            losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels, depth=3), logits=logits))
            tf.summary.scalar('loss', losses)

            # Compute evaluation metrics.
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=predicted_classes,
                                           name='accuracy_op')[1]
            metrics = {'accuracy': accuracy}
            # Write to tensorboard
            tf.summary.scalar('accuracy', accuracy)

            # return losses and train_op when training
            if self.config.get('learning_algorithm') == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.get('learning_rate'))
            elif self.config.get('learning_algorithm') == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.config.get('learning_rate'))
            elif self.config.get('learning_algorithm') == 'rms':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.get('learning_rate'))
            else:
                raise Exception('You must specify the learning algorithm.')
            train_op = optimizer.minimize(loss=losses, global_step=tf.train.get_global_step())
            tf.logging.info('Current learning_algorithm is {}, learning_rate is {}'.format(self.config.get('learning_algorithm'),
                                                                                           self.config.get(
                                                                                               'learning_rate')))
            return losses, train_op, predictions, accuracy

    def add_lstm_layer(self, inputs, length, layer_name):
        """
        Add one bidirectional layer.
        :param intputs: inputs.
        :param length: a batch of length.
        :return: concated lstm output.
        """
        with tf.variable_scope(str(layer_name)+'_layer'):
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
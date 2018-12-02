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
# @Time    : 1/12/18 12:21
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: crf_layer.py

# To add current dir to search path to prevent some errors
import sys, os

from tensorflow.python.ops import control_flow_ops

sys.path.append(os.getcwd())

import tensorflow as tf

class crf_layer(object):
    def __init__(self, inputs, sequence_lengths, transition_prob):
        self.inputs = inputs
        self.sequence_lengths = sequence_lengths
        self.transition_prob = transition_prob

    def crf_alpha_matrix(self):
        """
        Calculate the alpha matrix of crf.
        :return: a matrix of alpha
        """
        # split the first and the rest of the inputs in preparation for the forward algorithm.
        first_input = tf.slice(self.inputs, begin=[0, 0, 0], size=[-1, 1, -1])
        initial_state = tf.squeeze(first_input)

        # if max_length of inputs is 1, simply return first_inpu
        def _single_seq_fn():
            return first_input

        # else: do forward computation
        def _multi_seq_fn():
            rest_of_input = tf.slice(self.inputs, begin=[0, 1, 0], size=[-1, -1, -1])

            forward_cell = tf.contrib.crf.CrfForwardRnnCell(self.transition_prob)
            alphas, _ = tf.nn.dynamic_rnn(
                cell=forward_cell,
                inputs=rest_of_input,
                sequence_length=self.sequence_lengths,
                initial_state=initial_state,
                dtype=tf.float32
            )
            final_alpha = tf.concat([first_input, alphas], axis=1)
            return final_alpha

        max_sequence_length = max(self.sequence_lengths)
        return control_flow_ops.cond(pred=tf.equal(max_sequence_length, 1),
                               true_fn=_single_seq_fn,
                               false_fn=_multi_seq_fn)
    def crf_beta_matrix(self, time_major=False):
        """
        Compute the beta maxtrix of crf.
        :return:
        """
        def _reverse(input_, seq_length, seq_dim, batch_dim):
            return tf.reverse_sequence(
                input=input_, seq_lengths=seq_length,
                seq_dim=seq_dim, batch_dim=batch_dim
            )

        if time_major:
            time_dim = 0
            batch_dim = 1
        else:
            time_dim = 1
            batch_dim = 0

        inputs_reverse = _reverse(input_=self.inputs, seq_length=self.sequence_lengths,
                                  seq_dim=time_dim, batch_dim=batch_dim)

        first_input = tf.slice(inputs_reverse, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(first_input)

        def _single_seq_fn():
            return first_input

        def _multi_seq_fn():
            rest_of_input = tf.slice(inputs_reverse, [0, 1, 0], [-1, -1, -1])
            # Compute the alpha values in the forward algorithm in order to get the
            # partition function.
            backward_cell = tf.contrib.crf.CrfForwardRnnCell(self.transition_prob)
            betas, _ = tf.nn.dynamic_rnn(
                cell=backward_cell,
                inputs=rest_of_input,
                sequence_length=self.sequence_lengths - 1,
                initial_state=initial_state,
                dtype=tf.float32)
            final_betas = tf.concat([first_input, betas], axis=1)
            betas_reverse = _reverse(final_betas, seq_length=self.sequence_lengths,
                                     seq_dim=time_dim, batch_dim=batch_dim)
            return betas_reverse

        max_sequence_length = max(self.sequence_lengths)
        return control_flow_ops.cond(pred=tf.equal(max_sequence_length, 1),
                               true_fn=_single_seq_fn,
                               false_fn=_multi_seq_fn)

    def crf_log_norm(self):
        """
        Compute the log norm of crf.
        :return:
        """
        return tf.contrib.crf.crf_log_norm(inputs=self.inputs,
                                           sequence_length=self.sequence_lengths,
                                           transition_params=self.transition_prob)

    def crf_output_prob(self):
        """
        Return the normalized output probability.
        :return:
        """
        return tf.exp(tf.clip_by_value(
            self.crf_alpha_matrix() + self.crf_beta_matrix() - tf.expand_dims(tf.expand_dims(self.crf_log_norm(),
                                                                                             dim=-1), dim=-1),
            clip_value_min=10e-8, clip_value_max=10e8
        ))


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
# @Time    : 1/12/18 12:21
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: crf_layer.py

# To add current dir to search path to prevent some errors
import sys, os

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

sys.path.append('../')

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
        first_input = array_ops.slice(self.inputs, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(first_input)

        # if max_length of inputs is 1, simply return first_inpu
        def _single_seq_fn():
            return first_input

        # else: do forward computation
        def _multi_seq_fn():
            rest_of_input = array_ops.slice(self.inputs, [0, 1, 0], [-1, -1, -1])

            forward_cell = tf.contrib.crf.CrfForwardRnnCell(self.transition_prob)
            alphas, _ = tf.nn.dynamic_rnn(
                cell=forward_cell,
                inputs=rest_of_input,
                sequence_length=self.sequence_lengths-1,
                initial_state=initial_state,
                dtype=tf.float32
            )
            final_alpha = tf.concat([first_input, alphas], axis=1)
            return final_alpha

        max_sequence_length = array_ops.shape(self.inputs)[1]
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

        first_input = array_ops.slice(inputs_reverse, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(first_input)

        def _single_seq_fn():
            return first_input

        def _multi_seq_fn():
            rest_of_input = array_ops.slice(inputs_reverse, [0, 1, 0], [-1, -1, -1])
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

        max_sequence_length = array_ops.shape(self.inputs)[1]
        return control_flow_ops.cond(pred=tf.equal(max_sequence_length, 1),
                               true_fn=_single_seq_fn,
                               false_fn=_multi_seq_fn)

    def crf_log_norm(self):
        """
        Compute the log norm of crf.
        :return:
        """
        return tf.contrib.crf.crf_log_norm(inputs=self.inputs,
                                           sequence_lengths=self.sequence_lengths,
                                           transition_params=self.transition_prob)

    def crf_output_prob(self):
        """
        Return the normalized output probability.
        :return:
        """
        return tf.clip_by_value(tf.exp(
            self.crf_alpha_matrix() + self.crf_beta_matrix() - tf.expand_dims(tf.expand_dims(self.crf_log_norm(),
                                                                                             dim=-1), dim=-1)
        ), clip_value_min=10e-8, clip_value_max=1)


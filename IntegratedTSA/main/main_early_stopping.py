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
# @Time    : 3/12/18 09:38
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: main_early_stopping.py
import sys, os
sys.path.append('../')

from model.params_and_config import paramsAndConfig
from model.sa_sent_tensorflow_main import Sa_Sent_Tensorflow

import tensorflow as tf
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=2, type=int, help='Please input the batch size, 8 default.')
parser.add_argument('--epoch', default=1, type=int, help='Please specify the epoch, 100 default.')
parser.add_argument('--learning_rate', default=0.01, type=float, help='Please specify the learning rate, default 0.1')
parser.add_argument('--learning_algorithm', default='sgd', type=str, help='Please specify the learning algorithm, default sgd')

def _parse_dataset(example_proto):
    features = {
        'inputs': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
        'masks': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
    }
    context_feature = {
        'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        'length': tf.FixedLenFeature(shape=[], dtype=tf.int64)
    }
    # parse each sequence example
    context_parsed, sequence = tf.parse_single_sequence_example(example_proto, context_features=context_feature,
                                                                sequence_features=features)
    return sequence['inputs'], sequence['masks'], context_parsed['length'], [context_parsed['label']]

def main(argv):
    args = parser.parse_args(argv[1:])
    model = Sa_Sent_Tensorflow()

    model.param_and_config.config['learning_rate'] = args.learning_rate
    model.param_and_config.config['learning_algorithm'] = args.learning_algorithm

    train_dataset = tf.data.TFRecordDataset(model.param_and_config.config.get('train'))
    dev_dataset = tf.data.TFRecordDataset(model.param_and_config.config.get('dev'))
    test_dataset = tf.data.TFRecordDataset(model.param_and_config.config.get('test'))

    train_dataset = train_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=([None], [None], [], [None]))
    dev_dataset = dev_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=([None], [None], [], [None]))
    test_dataset = test_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=([None], [None], [], [None]))

    # define re-initialized iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    dev_init_op = iterator.make_initializer(dev_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    inputs, masks, lengths, labels = iterator.get_next()

    saver = tf.train.Saver()

    losses, train_op, predictions, accuracy = model.model_only(inputs=inputs, masks=masks, length=lengths,
                                                               labels=labels)
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        best_acc = 0
        for i in range(args.epoch):
            print('Epoch {}/{}'.format(i+1, args.epoch))
            sess.run(train_init_op)
            j = 0
            while True:
                try:
                    j = j+1
                    real_loss, _ = sess.run([losses, train_op])
                    if j % 10 == 0:
                        tf.logging.info('step {}: Loss is {}'.format(j, real_loss))
                except tf.errors.OutOfRangeError:
                    sess.run(dev_init_op)
                    # sess.run(train_init_op)
                    mean_accuracy = 0
                    while True:
                        try:
                            mean_accuracy = sess.run(accuracy)
                        except tf.errors.OutOfRangeError:
                            tf.logging.info('Epoch {}: Evaluation accrarcy is {}'.format(i, mean_accuracy))
                            if mean_accuracy >= best_acc:
                                tf.logging.info('New best score!')
                                tf.logging.info('Saving the model...')
                                saver.save(sess=sess, global_step=tf.train.get_global_step(),
                                           save_path=model.param_and_config.config.get('output_path')+'/checkpoint/model.ckpt')
                                best_acc = mean_accuracy
                            break
                    break

        # Test the model
        saver.restore(sess=sess, save_path=model.param_and_config.config.get('output_path')+'/checkpoint/model.ckpt')
        sess.run(test_init_op)
        mean_accuracy = 0
        while True:
            try:
                mean_accuracy = sess.run(accuracy)
            except tf.errors.OutOfRangeError:
                tf.logging.info('The test accuracy is {}'.format(mean_accuracy))
                break


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
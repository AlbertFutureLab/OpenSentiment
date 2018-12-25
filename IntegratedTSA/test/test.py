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
# @Time    : 28/11/18 15:07
# @Author  : Leiming Du
# @FileName: test.py
import sys, os
sys.path.append('../')

from model.params_and_config import paramsAndConfig
from model.keras_model import KerasSaSentTensorflow
from Preproessing.dataUtils import data_utils
import tensorflow as tf
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=1, type=int, help='Please input the batch size, 8 default.')
parser.add_argument('--epochs', default=1, type=int, help='Please specify the epoch, 100 default.')

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
    # return (sequence['inputs'], sequence['masks'], [context_parsed['length']]), tf.one_hot(
    #     indices=[context_parsed['label']], depth=3)
    return sequence['inputs'], sequence['masks'], context_parsed['length'], [context_parsed['label']]

def get_optimizer(opt_name):
    if opt_name == 'sgd':
        return tf.train.GradientDescentOptimizer(0.001)


def main(argv):
    args = parser.parse_args(argv[1:])
    params_and_config = paramsAndConfig()
    data_tools = data_utils()

    dic = data_tools.read_gzip_serialized_file(params_and_config.config.get('dic_path'))

    # build the dataset pipeline
    train_dataset = tf.data.TFRecordDataset(params_and_config.config.get('train'))
    dev_dataset = tf.data.TFRecordDataset(params_and_config.config.get('dev'))
    test_dataset = tf.data.TFRecordDataset(params_and_config.config.get('test'))

    train_dataset = train_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=([None], [None], [], [None]))

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    inputs, masks, lengths, labels = iterator.get_next()

    iterator = train_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()
    inputs, masks, hehe, labels = sess.run(next_element)
    print(inputs)

    real_inputs = [dic[int(v)] for v in inputs[0]]
    print("real_inputs: ", real_inputs)
    print('masks: ', masks)
    print('length: ', hehe)
    print('labels: ', labels)

    # i=0
    # sess.run(train_init_op)
    # while True:
    #     try:
    #         i = i+1
    #         real_inputs = sess.run(inputs)
    #         if i % 10 == 0:
    #             print('step {}: loss {}'.format(i, real_inputs))
    #     except tf.errors.OutOfRangeError:
    #         print('helo')



if __name__ == '__main__':
    main(sys.argv[1:])

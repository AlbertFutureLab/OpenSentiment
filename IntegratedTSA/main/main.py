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
# @Time    : 1/12/18 14:12
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: main.py

import sys, os
import tensorflow as tf
import argparse
sys.path.append('../')
from model.sa_sent_tensorflow_main import Sa_Sent_Tensorflow
from model.params_and_config import paramsAndConfig
from Preproessing.createTFrecordFile import CreateTFrecordData

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=16, type=int, help='Please input the batch size, 8 default.')
parser.add_argument('--epochs', default=20, type=int, help='Please specify the epoch, 100 default.')
parser.add_argument('--learning_rate', default=1, type=float, help='Please specify the learning rate, default 0.1')
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
    return (sequence['inputs'], sequence['masks'], context_parsed['length']), [context_parsed['label']]

def main(argv):
    args = parser.parse_args(argv[1:])

    # config and params, model, datatools
    data_fetch = CreateTFrecordData()
    my_model = Sa_Sent_Tensorflow()

    # set args parameters
    my_model.param_and_config['learning_rate'] = args.learning_rate
    my_model.param_and_config['learning_algorithm'] = args.learning_algorithm

    # build the dataset pipeline
    train_dataset = tf.data.TFRecordDataset(my_model.param_and_config.config.get('train'))
    dev_dataset = tf.data.TFRecordDataset(my_model.param_and_config.config.get('dev'))
    test_dataset = tf.data.TFRecordDataset(my_model.param_and_config.config.get('test'))

    # shuffle, paddding, batch, repeat operation.
    train_dataset = train_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=(([None], [None], []), [None]))
    dev_dataset = dev_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=(([None], [None], []), [None]))
    test_dataset = test_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=(([None], [None], []), [None]))

    # Build the model
    model = tf.estimator.Estimator(
        model_fn=my_model.my_model,
        model_dir=my_model.param_and_config.config.get('output_path'),
        config=tf.estimator.RunConfig(save_summary_steps=100, save_checkpoints_steps=1000)
    )

    # Train the model
    model.train(input_fn=train_dataset, steps=args.train_steps)

    # evaluate the model
    eval_result = model.evaluate(dev_dataset)
    tf.logging.info('The evaluation result is {accuracy:0.3f}...'.format(**eval_result))

    # Test the model
    test_result = model.evaluate(input_fn = test_dataset)
    tf.logging.info('The test result is {accuracy:0.3f}...'.format(**test_result))


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
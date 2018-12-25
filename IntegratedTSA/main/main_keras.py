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
# @Time    : 3/12/18 17:37
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: main_keras.py
import sys, os
sys.path.append('../')

from model.params_and_config import paramsAndConfig
from model.keras_model import KerasSaSentTensorflow
import tensorflow as tf
import numpy as np

import argparse

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
    return (sequence['inputs'], sequence['masks'], context_parsed['length']), tf.one_hot(indices=context_parsed['label'], depth=3)

def get_optimizer(opt_name, learning_rate):
    if opt_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif opt_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        raise Exception('You must specify a standard training algorithm.')

def log_likelihood(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int64)
    print('y_true, y_pred: ', y_true, y_pred)
    return tf.reduce_mean(tf.map_fn(lambda (x, y): -tf.log(tf.squeeze(x)[tf.squeeze(y)]), (y_pred, y_true), dtype=tf.float32))

def my_accuracy(y_true, y_pred):
    y_pred_label = tf.argmax(tf.squeeze(y_pred), axis=-1)
    y_true_label = tf.cast(tf.squeeze(y_true), dtype=tf.int64)
    print('y_pred_label, y_true', y_pred_label, y_true_label)
    return tf.reduce_mean(tf.cast(tf.equal(y_pred_label, y_true_label), dtype=tf.float32))

def category_modified_one_hot_acc(y_true, y_pred):
    y_true = tf.one_hot(indices=tf.cast(y_true, dtype=tf.int32), depth=3)
    return tf.keras.metrics.categorical_accuracy(y_true=y_true, y_pred=y_pred)

def main(argv):
    args = parser.parse_args(argv[1:])

    model = KerasSaSentTensorflow()
    model.param_and_config.config['learning_rate'] = args.learning_rate
    model.param_and_config.config['learning_algorithm'] = args.learning_algorithm

    # build the dataset pipeline
    train_dataset = tf.data.TFRecordDataset(model.param_and_config.config.get('train'))
    dev_dataset = tf.data.TFRecordDataset(model.param_and_config.config.get('dev'))
    test_dataset = tf.data.TFRecordDataset(model.param_and_config.config.get('test'))

    # shuffle, paddding, batch, repeat operation.
    train_dataset = train_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=(([None], [None], []), [None])).repeat()
    dev_dataset = dev_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=(([None], [None], []), [None])).repeat()
    test_dataset = test_dataset.map(map_func=_parse_dataset).shuffle(buffer_size=100).padded_batch(
        batch_size=args.batch_size, padded_shapes=(([None], [None], []), [None]))

    # create the model
    model = KerasSaSentTensorflow()
    tf.keras.backend.set_session(
        session=tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))
    tf.keras.backend.get_session().run(tf.global_variables_initializer())
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    optimizer = get_optimizer(model.param_and_config.config.get('learning_algorithm'), learning_rate=
                              model.param_and_config.config.get('learning_rate'))
    # model.compile(optimizer=optimizer,
    #               loss=log_likelihood,
    #               metrics=[category_modified_one_hot_acc])

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_acc'),
        # model checkpoint callback
        tf.keras.callbacks.ModelCheckpoint(filepath=model.param_and_config.config.get('output_path'),
                                           monitor='val_acc', save_best_only=True, mode='auto', period=1)
    ]

    model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=(5600/args.batch_size), validation_data=train_dataset, validation_steps=(5600/args.batch_size))

    model.evaluate(test_dataset, batch_size=args.batch_size)
    print model.predict(test_dataset, batch_size=args.batch_size)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
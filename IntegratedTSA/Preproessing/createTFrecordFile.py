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
# @Time    : 28/11/18 16:37
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: createTFrecordFile.py

# To add current dir to search path to prevent some errors
import os, time, sys
from tqdm import tqdm
reload(sys)
sys.setdefaultencoding('utf8') # To prevent any coding errores in python2

import math
import tensorflow as tf
import numpy as np

# To add current dir to search path to prevent some errors
sys.path.append('../')

QUEUE_CAPACITY = 500
SHUFFLE_MIN_AFTER_DEQUEUE = QUEUE_CAPACITY // 5

class CreateTFrecordData(object):
    def __init__(self):
        pass

    def _bytes_feature(self, value):
        """
        To change a list of value (format byte) to tf.train.Feature
        :param value: is or changed to list
        :return: tf.train.Feature
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_feature(self, value):
        """
        To change a list of value (format int64) to tf.train.Feature
        :param value: is or changed to list
        :return: tf.train.Feature
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(self, value):
        """
        To change a list of value (format float) to tf.train.Feature
        :param value: is or changed to list
        :return: tf.train.Feature
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def make_fixed_length_example(self, inputs, masks, label, max_senquence_length=100, padding_tag=0):
        """
        Given one sequence (usually a list of ids), return a tf.train.Example object.
        :param inputs: a list of ids (usually)
        :param masks: a list of masks
        :param label: the label of the sequence
        :param max_senquence_length: max length of the sequence (default 100)
        :param padding_tag: padding tag
        :return: a tf.train.Example file.
        """
        assert(len(inputs) == len(masks))

        # if the length of inputs exceeds the max_sequence_length, cut the inputs down.
        if len(inputs) > max_senquence_length:
            inputs = inputs[:(max_senquence_length-1)]
            masks = masks[:(max_senquence_length-1)]
        else:
            inputs.extend([padding_tag] * (max_senquence_length - len(inputs)))
            masks.extend([padding_tag] * (max_senquence_length - len(masks)))

        input_feature = self._int64_feature(value=inputs)
        mask_feature = self._int64_feature(value=masks)
        label_feature = self._int64_feature(value=label)
        length_feature = self._int64_feature(value=len(inputs))

        # get the feature collection list
        feature_collection = {
            'inputs': input_feature,
            'masks': mask_feature,
            'label': label_feature,
            'length': length_feature
        }

        return tf.train.Example(features=tf.train.Features(feature=feature_collection))

    def max_sequence_example(self, inputs, masks, label):
        """
        Given inputs, masks and label, return tf.train.SequenceExample object.
        :param inputs: a list of ids.
        :param masks: a list of masks.
        :param label: one label.
        :return: a tf.train.SequenceExample
        """
        assert(len(inputs) == len(masks))

        input_feature = [self._int64_feature(input_) for input_ in inputs]
        mask_feature = [self._int64_feature(mask_) for mask_ in masks]
        label_feature = self._int64_feature(label)
        length_feature = self._int64_feature(len(inputs))

        feature_collection = {
            'inputs': tf.train.FeatureList(feature=input_feature),  # the feature in tf.train.FeatureList should be a list
            'masks': tf.train.FeatureList(feature=mask_feature),
        }

        context_features = {
            'label': label_feature,
            'length': length_feature
        }

        return tf.train.SequenceExample(context=tf.train.Features(feature=context_features),
            feature_lists=tf.train.FeatureLists(feature_list=feature_collection))

    def sequence_serialized_to_file(self, filename, inputs, masks, labels, fixed=False, max_sentence_length=100):
        """
        Given several inputs and relevant features, write the serialized object into file.
        :param filename: The path to the serialized file.
        :param inputs: a list of lists of ids
        :param masks: a list of lists of masks
        :param labels: a list of lists of labels
        :param max_sentence_length: max length of the entire dataset.
        :return:
        """
        tf.logging.info('Totally have {} inputs.'.format(len(inputs)))
        writer = tf.python_io.TFRecordWriter(filename)

        tf.logging.info('Starting to write the serialized data...')
        if fixed:
            for i in tqdm(range(len(inputs)), desc='Processing dataset'):
                writer.write(self.make_fixed_length_example(inputs=inputs[i], masks=masks[i],
                                                            label=labels[i], max_senquence_length=max_sentence_length).SerializeToString())
        else:
            for i in tqdm(range(len(inputs)), desc='Processing dataset'):
                writer.write(self.max_sequence_example(inputs=inputs[i], masks=masks[i],
                                                            label=labels[i]).SerializeToString())
        writer.close()  # must close the writer, otherwise you cannot read the serialized file.
        tf.logging.info('Finish writing the serialized data. Totally wirte {} instances.'.format(len(inputs)))

    def get_padded_batch(self, file_list, batch_size, fixed=False, max_sequence_length=100, epoch=10000, num_enqueuing_thread=1, shuffle=True):
        """
        Read examples form a list of filename with 'epoch' epoches.
        :param file_list: a list of filenames.
        :param batch_size: The size of a batch
        :param epoch: the epoches of the training.
        :param num_enqueuing_thread: multithreads.
        :param shuffle: whether to shuffle, default True
        :return:
        """
        if not isinstance(file_list, list):
            file_list = [file_list]
        for single_file in file_list:
            if not os.path.exists(single_file):
                raise Exception('File {} does not exists.'.format(single_file))
        file_queue = tf.train.string_input_producer(file_list, num_epochs=epoch)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)

        if fixed:
            features = {
                'inputs': tf.FixedLenFeature(shape=[max_sequence_length], dtype=tf.int64),
                'masks': tf.FixedLenFeature(shape=[max_sequence_length], dtype=tf.int64),
                'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
                'length': tf.FixedLenFeature(shape=[], dtype=tf.int64)
            }
            # parse each example
            sequence = tf.parse_single_example(serialized_example, features=features)
            output_sequence = [sequence['inputs'], sequence['masks'], sequence['label'], sequence['length']]
        else:
            features = {
                'inputs': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
                'masks': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
            }
            context_feature = {
                'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
                'length': tf.FixedLenFeature(shape=[], dtype=tf.int64)
            }
            # parse each sequence example
            context_parsed, sequence = tf.parse_single_sequence_example(serialized_example, context_features=context_feature,
                                                        sequence_features=features)
            output_sequence = [sequence['inputs'], sequence['masks'], context_parsed['label'], context_parsed['length']]

        if shuffle:
            if num_enqueuing_thread < 2:
                raise ValueError("'num_enqueuing_thread' must be at least 2 when shuffling.")
            shuffle_threads = int(math.ceil(num_enqueuing_thread / 2.))

            min_after_dequeue = self.count_records(
                file_list, stop_at=SHUFFLE_MIN_AFTER_DEQUEUE
            )
            output_sequence = self._shuffle_inputs(output_sequence, capacity=QUEUE_CAPACITY,
                                                 min_after_dequeue=min_after_dequeue,
                                                 num_threads=shuffle_threads)
            num_enqueuing_thread -= shuffle_threads

        return tf.train.batch(
            output_sequence,
            batch_size=batch_size,
            capacity=QUEUE_CAPACITY,
            num_threads=num_enqueuing_thread,
            dynamic_pad=True,
            allow_smaller_final_batch=False
        )

    def count_records(self, file_list, stop_at=None):
        """
        Count number of records in files from file_list up to stop_at
        :param file_list: List of TFRecord files to count records in.
        :param stop_at: Optional number of records to stop counting in.
        :return: Integer number of records in files from file_list up to stop_at
        """
        num_records = 0
        for tfrecord_file in file_list:
            tf.logging.info('Counting records in {}'.format(tfrecord_file))
            for _ in tf.python_io.tf_record_iterator(tfrecord_file):
                num_records += 1
                if stop_at and num_records >= stop_at:
                    tf.logging.info('Number of records is at least {}'.format(num_records))
                    return num_records
        tf.logging.info('Total records: {}'.format(num_records))
        return num_records

    def _shuffle_inputs(self, input_tensors, capacity, min_after_dequeue, num_threads):
        """ Shuffle tensors in input_tensors, maintaining grouping"""
        shuffle_queue = tf.RandomShuffleQueue(
            capacity=capacity, min_after_dequeue=min_after_dequeue, dtypes=[t.dtype for t in input_tensors]
        )
        enqueue_op = shuffle_queue.enqueue(input_tensors)
        runner = tf.train.QueueRunner(shuffle_queue, [enqueue_op] * num_threads)
        tf.train.add_queue_runner(runner)

        output_tensors = shuffle_queue.dequeue()

        for i in xrange(len(input_tensors)):
            output_tensors[i].set_shape(input_tensors[i].shape)

        return output_tensors


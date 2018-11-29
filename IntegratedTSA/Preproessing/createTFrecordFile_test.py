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
# @Time    : 28/11/18 19:15
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: createTFrecordFile_test.py
import numpy as np
import tensorflow as tf
from createTFrecordFile import CreateTFrecordData

tf.logging.set_verbosity(tf.logging.INFO) # setting the info log visible

def main():
    create_tf_tool = CreateTFrecordData()
    tfrecord_filename = '../data/dataset_test/serialized_dataset.tfrecords'

    # Given inputs, create the dataset
    inputs = np.random.rand(300, 80).astype(int).tolist()
    masks = np.random.rand(300, 80).astype(int).tolist()
    label = np.random.rand(300).astype(int).tolist()

    create_tf_tool.sequence_serialized_to_file(filename=tfrecord_filename, inputs=inputs, masks=masks, labels=label)

    # Read tensors from tfrecord_filename
    read_inputs, read_masks, read_labels, read_lengths = create_tf_tool.get_padded_batch([tfrecord_filename],
                                                                                         batch_size=2, epoch=2,
                                                                                         shuffle=True, num_enqueuing_thread=3
                                                                                         )

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.local_variables_initializer())  # initializer the local variables
        sess.run(tf.global_variables_initializer())  # initializer the global variables

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # The pipeline would raise an exception when the pipeline ends.
        try:
            while not coord.should_stop():
                input_value, mask_value, label_value, length_value = \
                    sess.run([read_inputs, read_masks, read_labels, read_lengths])
                tf.logging.info('The input is {}, mask is {}, label is {}, length is {}'.format(
                    input_value, mask_value, label_value, length_value))
        except tf.errors.OutOfRangeError:
            tf.logging.info('Reading test finished!')
        coord.request_stop()
        coord.join(threads=threads)


if __name__=='__main__':
    main()

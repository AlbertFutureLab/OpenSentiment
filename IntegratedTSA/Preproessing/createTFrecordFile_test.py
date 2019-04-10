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
# @Time    : 28/11/18 19:15
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: createTFrecordFile_test.py

# To add current dir to search path to prevent some errors
import sys
import os
import numpy as np
import tensorflow as tf
from createTFrecordFile import CreateTFrecordData

tf.logging.set_verbosity(tf.logging.INFO) # setting the info log visible

# To add current dir to search path to prevent some errors
sys.path.append('../')

def main():
    create_tf_tool = CreateTFrecordData()
    tfrecord_filename = '../data/dataset_test/serialized_dataset.tfrecords'

    # Given inputs, create the dataset

    # ---- test fixed length ----
    # inputs = np.random.rand(300, 80).astype(int).tolist()
    # masks = np.random.rand(300, 80).astype(int).tolist()
    # label = np.random.rand(300).astype(int).tolist()

    # ---- test variable length ----
    inputs = [[1, 2, 31, 2], [2, 3, 1, 2, 3, 23, 31, 1], [1, 2]]
    masks = [[1, 2, 31, 2], [2, 3, 1, 2, 3, 23, 31, 1], [1, 2]]
    label = [1, 2, 1]

    create_tf_tool.sequence_serialized_to_file(filename=tfrecord_filename, inputs=inputs, masks=masks, labels=label, fixed=False)

    # Read tensors from tfrecord_filename
    read_inputs, read_masks, read_labels, read_lengths = create_tf_tool.get_padded_batch([tfrecord_filename],
                                                                                         batch_size=2, epoch=2,
                                                                                         shuffle=True, num_enqueuing_thread=3,
                                                                                         fixed=False
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

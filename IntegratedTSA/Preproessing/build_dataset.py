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
# @Time    : 2/12/18 10:15
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: build_dataset.py
import sys, os
sys.path.append('../')
import tensorflow as tf
from createTFrecordFile import CreateTFrecordData
from model.params_and_config import paramsAndConfig
from dataUtils import data_utils

def main():
    param_and_config = paramsAndConfig()
    tf_data_creation = CreateTFrecordData()
    data_tools = data_utils()

    # Generate the word dic and embedding file
    # data_tools.generate_dic_embedding(filename=param_and_config.config.get('original_embedding_path'),
    #                                   dic_path=param_and_config.config.get('dic_path'),
    #                                   final_embedding_path=param_and_config.config.get('embedding_path'))

    # Read the raw English tweets and write them to tfrecords
    raw_data_path = '../data/Tweets/English/raw'
    for data in ['train', 'dev', 'test']:
        inputs, masks, labels = data_tools.read_txt_file(filename=os.path.join(raw_data_path, data+'.raw.txt'),
                                 dic_path=param_and_config.config.get('dic_path'))
        tf_data_creation.sequence_serialized_to_file(filename=param_and_config.config.get(data),
                                                     inputs=inputs, masks=masks, labels=labels,
                                                     fixed=False, max_sentence_length=300)

    # Test whether the tfrecords is ok.
    for data in ['train', 'dev', 'test']:
        result = tf_data_creation.get_padded_batch(file_list=param_and_config.config.get(data), batch_size=2, epoch=1, shuffle=False)
        print(result)


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

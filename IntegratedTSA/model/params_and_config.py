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
# @Time    : 30/11/18 18:21
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: params_and_config.py

# To add current dir to search path to prevent some errors
import sys, os
sys.path.append('../')

class paramsAndConfig(object):
    def __init__(self):
        self.params = {
            # whether to use elmo as the input
            'elmo': False,
            # whether to use Bert as the input
            'bert': False,
            # whether to use word2vec/Glove embedding
            'word_embedding': True,
            # whetehr to use char embedding
            'char_embedding': False,

            # word_embedding && char_embedding config
            'word_dimension': 25,
            'char_dimension': 10,

            # mask dim
            'mask_dim': 10,

            # LSTM hidden units config
            'hidden_dimension': 300,
            # num of layers
            'layer_num': 3,
            # whether to use residual network
            'if_residual': True,
            # whether to use layer normalization
            'layer_norm': False,
            # whether to use dropout and dropout rate
            'if_dropout': True,
            'dropout_rate': 0.4,

            # logits config
            'n_classes': 3,
        }

        # config file
        self.config = {
            'output_path': '../data/Tweets/English/model/output',
            'dic_path': '../data/Tweets/English/embedding/dic.pkl',
            'original_embedding_path': '../data/Tweets/English/embedding/original_embedding.txt',
            'embedding_path': '../data/Tweets/English/embedding/embedding.pkl',
            'learning_rate': 0.001,
            'learning_algorithm': 'sgd',
            'train': '../data/Tweets/English/tfrecords_file/train.tfrecords',
            'dev': '../data/Tweets/English/tfrecords_file/dev.tfrecords',
            'test': '../data/Tweets/English/tfrecords_file/test.tfrecords',
            'dataset_path': '../data/tweets/english'
        }

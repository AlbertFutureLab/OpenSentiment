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
# @Time    : 28/11/18 14:41
# @Author  : Leiming Du
# @FileName: dataUtils.py

import os, time, sys
reload(sys)
sys.setdefaultencoding('utf8') # To prevent any coding errores in python2

import cPickle as pickle
import gzip
import numpy as np

class data_utils(object):
    def __init__(self):
        pass

    def generate_dic_embedding(self, filename, dic_path, final_embedding_path, memory_first=False):
        """
        Read the large embedding file and generate the serialized dic file and embedding file.
        :param pathname: The path to the large embedding file.
        :param memory_first: choose saving-memory first or speed first. speed fist as default.
        :return:
        """
        if not os.path.exists(path=filename):
            raise Exception('The file does not exist!')

        # Read file and generate dic and final embedding
        word2id = []
        embedding_to_serialized = []

        if not memory_first:
            with open(filename, 'r') as fr:
                original_embedding = fr.readlines()
                for single_line in original_embedding:
                    single_line_split = single_line.strip().split()
                    word2id.append(single_line_split[0])
                    embedding_to_serialized.append(single_line_split[1:])
        else:
            original_embedding = np.loadtxt(filename, dtype=str)
            word2id = original_embedding[:, 0]
            # embedding_to_serialized = np.delete(original_embedding, 0, 1).astype(float) # np.delete is time-comsuming as it would copy the array
            embedding_to_serialized = original_embedding[:, 1:].astype(float)

        # Serialize the dic and final embedding_path to disk
        self.write_gzip_serialized_file(serialized_object=word2id, filename=dic_path)
        self.write_gzip_serialized_file(serialized_object=embedding_to_serialized, filename=final_embedding_path)

    def read_gzip_serialized_file(self, filename):
        """
        Read the serialized file and return the object.
        :param filename: The path to the file
        :return: the object
        """
        if not os.path.exists(path=filename):
            raise Exception('The file does not exist!')

        with gzip.open(filename, 'rb') as fr:
            returned_object = pickle.load(fr)
        return returned_object

    def write_gzip_serialized_file(self, serialized_object, filename):
        """
        Write the compressed serialized object to filename.
        :param filename: The path to file.
        :return:
        """
        if os.path.exists(filename):
            print('Warning!: The existed the same file.')

        with gzip.open(filename, 'wb') as fw:
            pickle.dump(serialized_object, fw)

        print('Successfully write the serialized object to the file [{}]!'.format(filename))

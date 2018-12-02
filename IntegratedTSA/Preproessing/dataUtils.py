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
# @Email   : duleimingdo@gmail.com
# @FileName: dataUtils.py
from collections import namedtuple
import os, time, sys
reload(sys)
sys.setdefaultencoding('utf8') # To prevent any coding errores in python2

import cPickle as pickle
import gzip
import numpy as np
import codecs

# To add current dir to search path to prevent some errors
sys.path.append('../')

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
                single_original_embedding = fr.readline()
                while single_original_embedding:
                    single_line_split = single_original_embedding.strip().split()
                    word2id.append(single_line_split[0])
                    embedding_to_serialized.append(single_line_split[1:])
                    single_original_embedding = fr.readline()
        else:
            original_embedding = np.loadtxt(filename, dtype=str)
            word2id = original_embedding[:, 0]
            # embedding_to_serialized = np.delete(original_embedding, 0, 1).astype(float) # np.delete is time-comsuming as it would copy the array
            embedding_to_serialized = original_embedding[:, 1:].astype(float)

        # Serialize the dic and final embedding_path to disk
        self.write_gzip_serialized_file(serialized_object=word2id, filename=dic_path)

        # Here, using np.savetxt may have a better disk usage instead of gzip.serialized.
        np.savetxt(fname=final_embedding_path, X=embedding_to_serialized)
        # self.write_gzip_serialized_file(serialized_object=embedding_to_serialized, filename=final_embedding_path)

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

    def read_txt_file(self, filename, dic_path):
        """
        Read the txt tweets file. Here is the format of the file:
        First line: tweets or sentence.
        Second line: keywords.
        Third line: label.
        :param filename: The path to file
        :return: (texts(ids), masks, labels) tuple
        """
        # Build the Sentence Instance
        SenInst = namedtuple('Instance', 'text keyword label mask')
        instance_lists = []

        with codecs.open(filename, 'r', encoding='utf8') as fr:
            text = fr.readline().strip().lower()
            keyword = fr.readline().strip().lower()
            label = fr.readline().strip()
            while text:
                if label == '-1':  # Negative
                    real_label = 0
                elif label == '0':  # Netrual
                    real_label = 1
                elif label == '1':  # Positive
                    real_label = 2
                else:
                    raise Exception('Something wrong when reading the label.')
                text = text.split()
                keyword = keyword.split()
                keyword_location = text.index(u'$t$')

                # Create mask vector
                mask = [0]*keyword_location
                mask.extend([1]*len(keyword))
                mask.extend([0]*(len(text)-1-keyword_location))

                # Create new text
                new_text = text[:keyword_location]
                new_text.extend(keyword)
                new_text.extend(text[(keyword_location+1):])

                # add the instance to list
                instance_lists.append(SenInst(text=new_text, keyword=keyword, label=real_label, mask=mask))

                text = fr.readline().strip().lower()
                keyword = fr.readline().strip().lower()
                label = fr.readline().strip()
        dic_list = self.read_gzip_serialized_file(dic_path)
        word2id = {k: v for v, k in enumerate(dic_list)}

        inputs = []
        masks = []
        labels = []
        for single_instance in instance_lists:
            inputs.append(self.text_to_ids(single_instance.text, word2id=word2id))
            masks.append(single_instance.mask)
            labels.append(single_instance.label)
        return inputs, masks, labels

    def text_to_ids(self, text, word2id):
        """
        Give a list of words and dic, change it to a list of ids.
        :param text: a list of words.
        :param word2id: a dict where key: word, value: id
        :return: a list of ids.
        """
        return [word2id.get(x) if word2id.get(x) else word2id.get('unk') for x in text]

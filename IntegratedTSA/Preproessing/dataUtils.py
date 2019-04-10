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
unk = '$UNK$'

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
        word2id = [unk]
        embedding_to_serialized = [np.random.uniform(-0.00001, 0.00001, 100).astype(float)]

        if not memory_first:
            with open(filename, 'r') as fr:
                single_original_embedding = fr.readline()
                while single_original_embedding:
                    single_line_split = single_original_embedding.strip().split()
                    word2id.append(single_line_split[0])
                    embedding_to_serialized.append(single_line_split[1:])
                    single_original_embedding = fr.readline()
            embedding_to_serialized = np.array(embedding_to_serialized).astype(float)
        else:
            original_embedding = np.loadtxt(filename, dtype=str)
            word2id.extend(original_embedding[:, 0])
            # embedding_to_serialized = np.delete(original_embedding, 0, 1).astype(float) # np.delete is time-comsuming as it would copy the array
            embedding_to_serialized.extend(original_embedding[:, 1:].astype(float))

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
            print('Warning!: There existed the same file.')

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


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
    data_tools.generate_dic_embedding(filename=param_and_config.config.get('original_embedding_path'),
                                      dic_path=param_and_config.config.get('dic_path'),
                                      final_embedding_path=param_and_config.config.get('embedding_path'))

    # Read the raw English tweets and write them to tfrecords
    raw_data_path = '../data/Tweets/English/raw'
    for data in ['train', 'dev', 'test']:
        inputs, masks, labels = data_tools.read_txt_file(filename=os.path.join(raw_data_path, data+'.raw.txt'),
                                 dic_path=param_and_config.config.get('dic_path'))
        tf_data_creation.sequence_serialized_to_file(filename=param_and_config.config.get(data),
                                                     inputs=inputs, masks=masks, labels=labels,
                                                     fixed=False, max_sentence_length=100)

    # Test whether the tfrecords is ok.
    for data in ['train', 'dev', 'test']:
        result = tf_data_creation.get_padded_batch(file_list=param_and_config.config.get(data), batch_size=2, epoch=1, shuffle=False)
        print(result)


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

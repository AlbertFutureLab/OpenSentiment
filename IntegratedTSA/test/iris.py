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
# @Time    : 6/12/18 14:53
# @Author  : Leiming Du
# @Email   : duleimingdo@gmail.com
# @FileName: iris.py
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras

class MyModel(keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    self.dense_1 = tf.layers.Dense(units=16, dtype=tf.float64)
    self.dense_2 = tf.layers.Dense(units=num_classes, activation='softmax', dtype=tf.float64)
    # Define your layers here.
    # self.x, self.y = inputs
  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    # x, y = inputs
    # x = self.dense_1(x + y)
    x = self.dense_1(inputs)
    x = self.dense_2(x)
    return x

#Data download and dataset creation witout tf.data
train_ds_url = "http://download.tensorflow.org/data/iris_training.csv"
test_ds_url = "http://download.tensorflow.org/data/iris_test.csv"
ds_columns = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Plants']
species = np.array(['Setosa', 'Versicolor', 'Virginica'], dtype=np.object)

#Load data
categories='Plants'
train_path = tf.keras.utils.get_file(train_ds_url.split('/')[-1], train_ds_url)
test_path = tf.keras.utils.get_file(test_ds_url.split('/')[-1], test_ds_url)
train = pd.read_csv(train_path, names=ds_columns, header=0)
train_plantfeatures, train_categories = train, train.pop(categories)
test = pd.read_csv(test_path, names=ds_columns, header=0)
test_plantfeatures, test_categories = test, test.pop(categories)

#to_categorical
y_categorical = tf.contrib.keras.utils.to_categorical(train_categories, num_classes=3)
y_categorical_test = tf.contrib.keras.utils.to_categorical(test_categories, num_classes=3)

y_categorical = tf.cast(y_categorical, tf.float64)
y_categorical_test = tf.cast(y_categorical_test, tf.float64)

# Build the dataset
# train
dataset = tf.data.Dataset.from_tensor_slices((train_plantfeatures, y_categorical))
dataset = dataset.batch(32)
dataset = dataset.shuffle(1000)
dataset = dataset.repeat()

#test
dataset_test = tf.data.Dataset.from_tensor_slices((test_plantfeatures, y_categorical_test))
dataset_test = dataset_test.batch(32)
dataset_test = dataset_test.shuffle(1000)
dataset_test = dataset_test.repeat()

print(dataset)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(16, input_dim=4),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax),
])

# model = MyModel(num_classes=3)
# eager off
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])

# model.summary()

# #eager on
opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(dataset, steps_per_epoch=10, epochs=1, verbose=1, validation_data=dataset_test, validation_steps=1000)
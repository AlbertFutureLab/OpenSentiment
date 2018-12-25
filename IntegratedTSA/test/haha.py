from tensorflow import keras
import tensorflow as tf
import numpy as np
class MyModel(keras.Model):

  def __init__(self, num_classes=10, *args, **kwargs):
    print(args, kwargs)
    super(MyModel, self).__init__(name='my_model', *args, **kwargs)
    self.num_classes = num_classes
    self.dense_1 = tf.layers.Dense(units=30, activation='relu')
    self.dense_2 = tf.layers.Dense(units=num_classes, activation='softmax')

    # Define your layers here.
    # self.x, self.y = inputs
  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x, y = inputs
    x = self.dense_1(x + y)
    # x = self.dense_1(inputs)
    x = self.dense_2(x)
    return x

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)

data = np.random.random((100, 30)).astype(np.float32)
data = (data, data)
label = np.random.random((100, 10)).astype(np.float32)

val_data = np.random.random((50, 30)).astype(np.float32)
val_data = (val_data, val_data)
val_label = np.random.random((50, 10)).astype(np.float32)


train_dataset = tf.data.Dataset.from_tensor_slices((data, label)).batch(100).repeat()
train_dataset_1 = tf.data.Dataset.from_tensor_slices((data, label)).batch(100).repeat()
dev_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label)).batch(50).repeat()

inputinputs = (tf.keras.Input(shape=[None, 30], name='x'), tf.keras.Input(shape=[None, 30], name='y'))

tf.keras.backend.set_session(session=tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))
model = MyModel(num_classes=10)

model.compile(optimizer=tf.train.GradientDescentOptimizer(0.5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.built)


model.fit(train_dataset, epochs=200, steps_per_epoch=1000, validation_data=train_dataset, validation_steps=1000)


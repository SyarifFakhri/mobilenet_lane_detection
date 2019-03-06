import mobileNet_v3
import tensorflow as tf

file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

with tf.contrib.slim.arg_scope(mobileNet_v3.training_scope(is_training=True)):
    logits, endpoints = mobileNet_v3.mobilenet(images, 6)

#getting the variables except for the last layer
print(endpoints)
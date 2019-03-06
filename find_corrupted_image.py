'''Reads through all the images in a folder one by one to find a corrupted image'''

import os
import tensorflow as tf
from dataHelperFunctions import filterImages

dataset_path = "./Full dataset - normalized/"
all_images = os.listdir(dataset_path)
all_images = filterImages(all_images)


name = tf.placeholder(tf.string)
image_string = tf.read_file(name)
image_decoded = tf.image.decode_jpeg(image_string)

with tf.Session() as sess:
    for imageName in all_images:
        print(imageName)
        read_image = sess.run(image_decoded, feed_dict={name:dataset_path + imageName})

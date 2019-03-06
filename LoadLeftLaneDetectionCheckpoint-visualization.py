import tensorflow as tf
import mobileNet_v3
import os
from random import shuffle
import cv2
from dataHelperFunctions import filterImages, extractLabels, addDatasetPath, drawOneLane
import math


def unnormalizeFromParams(output, norm_params):
    """from a normalized element it unnormalizes it back again so that we can interpret the result"""
    # for mean in norm_params[0:3]: #take the first three because that's the mean - changed norm params formatting
    for i in range(len(output)):
        output[i] = (float(output[i]) * float(norm_params[i*2 + 1])) + float(norm_params[i*2]) #multiply by the std deviation and add the mean
    return output


# dataset_path = "./UTKFace/Validation/"
dataset_path = "./0110 - dataset normalized/"
validation_path = "Validation/"
normalized_dataset = True
normalized_file_name = dataset_path + "NormalizationParams.txt"
root_folder = "./LeftLaneOnlyFolders/"
checkpoint = root_folder + "LANE_finetune_Checkpoint/FineTuneCheckpoint"
output_regression_count = 3

tf.reset_default_graph()

# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
file_input = tf.placeholder(tf.string, ())
image = tf.image.decode_jpeg(tf.read_file(file_input))
images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128. - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

# Note: arg_scope is optional for inference.
with tf.contrib.slim.arg_scope(mobileNet_v3.training_scope(is_training=False)):
    logits, endpoints = mobileNet_v3.mobilenet(images, 3)

# Restore using exponential moving average since it produces (1.5-2%) higher
# accuracy
# ema = tf.train.ExponentialMovingAverage(0.999)
# vars = ema.variables_to_restore()

# load the normalization params if the data is normalized
if normalized_dataset:
    text_file = open(normalized_file_name, "r")
    normalization_params = text_file.read().split()
    text_file.close()

saver = tf.train.Saver()
file_names = os.listdir(dataset_path + validation_path)
file_names = filterImages(file_names)
file_names = addDatasetPath(dataset_path + validation_path, file_names)
shuffle(file_names)
labels = extractLabels(file_names)

with tf.Session() as sess:
    saver.restore(sess, checkpoint)
    for i in range(len(file_names)):
        x = logits.eval(feed_dict={file_input: file_names[i]})
        print(x)
        x = x[0]
        unnormalizeFromParams(x, normalization_params)
        image = cv2.imread(file_names[i])

        image = drawOneLane(image, x[0], x[1], x[2], "blue")
        label = unnormalizeFromParams(labels[i], normalization_params)
        image = drawOneLane(image, label[0], label[1], label[2], "green")

        print("predicted", x)
        print("actual: ", label)
        cv2.imshow("image", image)
        cv2.waitKey(0)

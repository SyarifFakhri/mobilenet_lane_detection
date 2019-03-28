import tensorflow as tf
import mobileNet_v3
import os
from random import shuffle
import cv2
from dataHelperFunctions import filterImages, extractLabels, addDatasetPath, drawOneLaneFromPoints
import math
import numpy as np

def unnormalizeFromParams(output, norm_params):
    """from a normalized element it unnormalizes it back again so that we can interpret the result"""
    assert (len(norm_params) / len(output)) == 2, "There must be two norm params for every output!" + str(len(norm_params)) + " " + str(len(output))

    for i in range(len(output)):
        output[i] = (float(output[i]) * float(norm_params[i*2 + 1])) + float(norm_params[i*2]) #multiply by the std deviation and add the mean

    return output


# dataset_path = "./UTKFace/Validation/"
video_path = "D:/dataset/1. Dashcam/Dec 6/EVT1_20171207_100656.avi"


normalized_dataset = True
normalized_file_name = "./d_aug_two_lanes_percentage_dataset/NormalizationParams.txt"
root_folder = "./f_aug_two_lanes_percentage/"
checkpoint = root_folder + "LANE_finetune_Checkpoint/FineTuneCheckpoint"
output_regression_count = 46

tf.reset_default_graph()

# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
X = tf.placeholder(tf.float32, [1, None, None, 3])
# images = tf.expand_dims(X, 0)
images = tf.cast(X, tf.float32) / 128. - 1
images = tf.image.resize_images(images, (224, 224))

# Note: arg_scope is optional for inference.
with tf.contrib.slim.arg_scope(mobileNet_v3.training_scope(is_training=False)):
    logits, endpoints = mobileNet_v3.mobilenet(images, output_regression_count)

# Restore using exponential moving average since it produces (1.5-2%) higher
# accuracy
# ema = tf.train.ExponentialMovingAverage(0.999)o
# vars = ema.variables_to_restore()

# load the normalization params if the data is normalized
if normalized_dataset:
    text_file = open(normalized_file_name, "r")
    normalization_params = text_file.read().split()
    text_file.close()

saver = tf.train.Saver()
# shuffle(file_names)
src = cv2.VideoCapture(video_path)

yPoints = []
for i in range(0, 224, 10):
    yPoints.append(float(i))

with tf.Session() as sess:
    saver.restore(sess, checkpoint)

    while True:
        _, src_frame = src.read()
        frame_copy = cv2.resize(src_frame, (224, 224))

        # feed_input = cv2.cvtColor(feed_input, cv2.COLOR_BGR2RGB)
        feed_input = np.array(frame_copy)

        feed_input = feed_input.reshape((1, feed_input.shape[0], feed_input.shape[1], 3))
        x = logits.eval(feed_dict={X: feed_input})
        print(x)
        xPoints = x[0]

        unnormalizeFromParams(xPoints, normalization_params)


        xRightPoints = xPoints[0:23]
        # image = drawOneLaneFromPoints(image, x[0], x[1], x[2], "blue")
        # image = drawOneLaneFromPoints(image, x[3], x[4], x[5], "yellow")
        image = drawOneLaneFromPoints(frame_copy, xRightPoints, yPoints, "blue")

        xLeftPoints = xPoints[23:]
        image = drawOneLaneFromPoints(frame_copy, xLeftPoints, yPoints, "green")
        # label = unnormalizeFromParams(labels[i], normalization_params)
        # image = drawOneLaneFromPoints(image, label[0], label[1], label[2], "green")
        # image = drawOneLaneFromPoints(image, label[3], label[4], label[5], "red")

        print("predicted", x)
        # print("actual: ", label)
        cv2.imshow("image", image)
        cv2.waitKey(20)

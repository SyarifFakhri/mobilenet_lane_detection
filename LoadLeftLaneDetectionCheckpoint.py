# import tensorflow as tf
# import tensorflow_hub
# import cv2
# import numpy as np
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('./checkpoints/chk.meta')
#     saver.restore(sess, './checkpoints/chk')
#     graph = tf.get_default_graph()
#     inp = graph.get_tensor_by_name("Placeholder:0")
#     output = graph.get_tensor_by_name("final_result:0")
#
#     frame = cv2.imread("./dataset/bird.jpg")
#     frameCopy = frame.copy()
#     # frameCopy = np.array((frameCopy - np.min(frameCopy)) / (np.max(frameCopy) - np.min(frameCopy)))
#     frameCopy = np.array(frameCopy / 255)
#     # print(Input_image_shape.)
#
#     feed_input = cv2.resize(frameCopy, (224, 224))
#     feed_input = feed_input.reshape((1, feed_input.shape[0], feed_input.shape[1], 3))
#
#     print(sess.run(output,feed_dict={inp:feed_input}))
#

# import tensorflow as tf
#
# #Step 1
# #import the model metagraph
# saver = tf.train.import_meta_graph('./checkpoints/culane-train-model.ckpt.meta', clear_devices=True)
# #make that as the default graph
# graph = tf.get_default_graph()
# input_graph_def = graph.as_graph_def()
# sess = tf.Session()
# #now restore the variables
# saver.restore(sess, "./checkpoints/culane-train-model")
#
# #Step 2
# # Find the output name
# graph = tf.get_default_graph()
# for op in graph.get_operations():
#   print (op.name)
#
# #Step 3
# from tensorflow.python.platform import gfile
# from tensorflow.python.framework import graph_util
#
# output_node_names="final_result"
# output_graph_def = graph_util.convert_variables_to_constants(
#         sess, # The session
#         input_graph_def, # input_graph_def is useful for retrieving the nodes
#         output_node_names.split(",")  )
#
# #Step 4
# #output folder
# output_fld ='./'
# #output pb file name
# output_model_file = 'test.pb'
# from tensorflow.python.framework import graph_io
# #write the graph
# graph_io.write_graph(output_graph_def, output_fld, output_model_file, as_text=False)



import tensorflow as tf
import mobileNet_v3
import os
from random import shuffle
import cv2
from dataHelperFunctions import filterImages, extractLabels, addDatasetPath

def unnormalizeFromParams(output, norm_params):
    """from a normalized element it unnormalizes it back again so that we can interpret the result"""
    # for mean in norm_params[0:3]: #take the first three because that's the mean
    output[0] = (float(output[0]) * float(norm_params[3])) + float(norm_params[0])
    output[1] = (float(output[1]) * float(norm_params[4])) + float(norm_params[1])
    output[2] = (float(output[2]) * float(norm_params[5])) + float(norm_params[2])

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

#load the normalization params if the data is normalized
if normalized_dataset == True:
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
        unnormalizeFromParams(x[0], normalization_params)

        print("predicted", x)
        print("actual: ", unnormalizeFromParams(labels[i], normalization_params))
        cv2.imshow("image", cv2.imread(file_names[i]))
        cv2.waitKey(0)

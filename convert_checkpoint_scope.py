import tensorflow as tf
# from dataHelperFunctions import print_tensors_in_checkpoint_file
# import mobileNet_v3
#
# rootFolder = "./aug_two_lanes_v2/"
# CHECKPOINT_NAME = rootFolder + "LANE_finetune_Checkpoint/FineTuneCheckpoint"
# checkpoint = CHECKPOINT_NAME
#
# # with tf.Graph().as_default(), tf.Session().as_default() as sess:
# #   with tf.variable_scope('my-first-scope'):
# #     NUM_IMAGE_PIXELS = 784
# #     NUM_CLASS_BINS = 10
# #     x = tf.placeholder(tf.float32, shape=[None, NUM_IMAGE_PIXELS])
# #     y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASS_BINS])
# #
# #     W = tf.Variable(tf.zeros([NUM_IMAGE_PIXELS,NUM_CLASS_BINS]))
# #     b = tf.Variable(tf.zeros([NUM_CLASS_BINS]))
# #
# #     y = tf.nn.softmax(tf.matmul(x,W) + b)
# #     cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# #     saver = tf.train.Saver([W, b])
# #   sess.run(tf.global_variables_initializer())
# #   saver.save(sess, 'my-model')
#
# varsToIgnore = []
# #first get the list of names of all ops in the checkpoint
# varList = print_tensors_in_checkpoint_file(checkpoint, varsToIgnore)
# print(varList)
# file_input = tf.placeholder(tf.string, ())
#
# #dummy placeholder, this doesn't actually matter that much
# image = tf.image.decode_jpeg(tf.read_file(file_input))
# images = tf.expand_dims(image, 0)
# images = tf.cast(images, tf.float32) / 128.  - 1
# images.set_shape((None, None, None, 3))
# images = tf.image.resize_images(images, (224, 224))
#
# dummy_placeholder = tf.placeholder
# with tf.contrib.slim.arg_scope(mobileNet_v3.training_scope(is_training=True)):
#     logits, endpoints = mobileNet_v3.mobilenet(images, 6)
#
# print(len(varList))
# #convert it to a tf variable
# vars = []
# for name in varList:
#     vars = vars + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
# print(vars)

rootFolder = "./aug_two_lanes_v2/"
old_checkpoint = rootFolder + "LANE_finetune_Checkpoint/FineTuneCheckpoint"

new_checkpoint = rootFolder + "LANE_finetune_Checkpoint/new_scope/FineTuneCheckpoint_new_scope"

#ssd_mobilenet_checkpoint = "D:/models-master/research/object_detection/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt"

vars = tf.contrib.framework.list_variables(old_checkpoint)
print(len(vars))
print(vars)

#ssd_vars = tf.contrib.framework.list_variables(ssd_mobilenet_checkpoint)

with tf.Graph().as_default(), tf.Session().as_default() as sess:

  new_vars = []
  for name, shape in vars:
    v = tf.contrib.framework.load_variable(old_checkpoint, name)
    new_vars.append(tf.Variable(v, name=name.replace('MobilenetV2', 'FeatureExtractor/MobilenetV2')))

  print("new mobilenet: ",new_vars)
  print(len(new_vars))

  # print(ssd_vars)
  # new_vars = []
  # for name, shape in ssd_vars:
  #   v = tf.contrib.framework.load_variable(ssd_mobilenet_checkpoint, name)
  #   new_vars.append(tf.Variable(v))
  # print("ssd vars: ",new_vars)
  assert (len(new_vars) == len(vars)), "the new variable list is different from the old one. "

  saver = tf.train.Saver(new_vars)
  sess.run(tf.global_variables_initializer())
  saver.save(sess, new_checkpoint)



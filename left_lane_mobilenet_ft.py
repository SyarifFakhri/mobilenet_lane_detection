import tensorflow as tf
import mobileNet_v3
from tensorflow.python import pywrap_tensorflow
import os
import random
from dataHelperFunctions import prepare_file_system, print_tensors_in_checkpoint_file, extractLabels, addDatasetPath, filterImages

#define global variables here
learningRate = 0.0001
height = 224
width = 224
outputRegressionCount = 3
rootFolder = "./LeftLaneOnlyFolders/"
checkpoint = "./MobileNetCheckpoint/mobilenet_v2_1.0_224.ckpt"
EPOCHS = 10000
dataset_path = "./0110 - dataset normalized/"
BATCH_SIZE = 64
summaries_dir = rootFolder + "Summary_FineTune_mobilenet_LEFT_ONLY"
validation_path = dataset_path + "Validation/"
percentValidation = 0.1
CHECKPOINT_NAME = rootFolder + "LANE_finetune_Checkpoint/FineTuneCheckpoint"
global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

loadFromBaseMobilenet = False

if loadFromBaseMobilenet == False:
    checkpoint = CHECKPOINT_NAME


# def prepare_file_system(summaries_dir):
# 	# Set up the directory we'll write summaries to for TensorBoard
# 	if tf.gfile.Exists(summaries_dir):
# 		tf.gfile.DeleteRecursively(summaries_dir)
# 	tf.gfile.MakeDirs(summaries_dir)
# 	return

# def print_tensors_in_checkpoint_file(file_name, varsToIgnore):
#     """Gets the variable list with the excpetion of some variables
#     Inputs:
#         file_name
#         varsToIgnore: The vars to ignore, can be a list or a string
#     Outputs:
#         varlist: the list of variables
#     """
#     varlist=[]
#     reader = pywrap_tensorflow.NewCheckpointReader(file_name)
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     for key in sorted(var_to_shape_map):
#         if key in varsToIgnore:
#             print("skipping key:", key)
#             continue
#         varlist.append(key)
#     return varlist

def constructTrainingParams(inputPlaceholder, logit, ground_truth):
    """creates and returns the training optimizers
    Inputs:
        Input Placeholder: the input to enter
        Regression outputs: num of regression to output
    Outputs:
        trainStep/groundTruthInput/MSE_Loss"""
    loss = tf.reduce_mean(tf.squared_difference(logit, ground_truth))
    mae = tf.reduce_mean(tf.abs(tf.subtract(logit, ground_truth)))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op,loss, mae

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    # images = tf.expand_dims(image, 0)
    image_decoded = tf.image.decode_jpeg(image_string)
    # image_decoded = tf.expand_dims(image_decoded, 0)
    images_norm = tf.cast(image_decoded, tf.float32) / 128. - 1  # normalizing by dividing by mean
    images_norm.set_shape((None, None, 3))
    image_resized = tf.image.resize_images(images_norm, [height, width])
    return image_resized, label

# def extractLabels(pictureFileNames):
#     temp_labels = []
#     for name in pictureFileNames:
#         splitFiles = os.path.splitext(name)
#         name = splitFiles[0]
#         # do_label_filter = True
#         text_file = open(name + ".txt", "r")
#         lines = text_file.read().split()
#         temp_labels.append(lines)
#         text_file.close()
#     return temp_labels

# def addDatasetPath(datasetPath, iterable):
#     for i in range(len(iterable)):
#         iterable[i] = datasetPath + iterable[i]
#     return iterable

# def filterImages(array):
#     newArr = []
#     for name in array:
#         listName = os.path.splitext(name)
#         extension = listName[-1]
#         if extension == ".jpg":
#             newArr.append(name)
#     return newArr

###Begin main function here###
# create an input tensor
prepare_file_system(summaries_dir)
#load the data set
# A vector of filenames.
image_files = os.listdir(dataset_path)
image_files = filterImages(image_files)

random.shuffle(image_files)
#make the dataset the full path
image_files = addDatasetPath(dataset_path, image_files)
#files variable will be left with only the train ground truth
train_groundTruth = extractLabels(image_files)

#load validation fileset
validation_files = os.listdir(validation_path)
validation_files = filterImages(validation_files)
validation_files = addDatasetPath(validation_path, validation_files)
random.shuffle(validation_files)
validation_groundTruth = extractLabels(validation_files)

assert len(train_groundTruth) == len(image_files)
assert len(validation_groundTruth) == len(validation_files)

filenames_placeholder = tf.placeholder(tf.string)
# filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg"])
# `labels[i]` is the label for the image in `filenames[i].
labels_inputPlaceholder = tf.placeholder(tf.float32)

#Create a tf dataset that will batch the images to feed into training
batch_size = tf.placeholder(tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((filenames_placeholder, labels_inputPlaceholder))
dataset = dataset.map(_parse_function)
dataset = dataset.batch(batch_size).repeat() #order matters
iter = dataset.make_initializable_iterator()
input_images, labels = iter.get_next()

varsToIgnore = []
if loadFromBaseMobilenet == True:
    #load mobile net
    varsToIgnore = ["MobilenetV2/Logits/Conv2d_1c_1x1/biases", "MobilenetV2/Logits/Conv2d_1c_1x1/weights"]

#first get the list of names of all ops in the checkpoint
varList = print_tensors_in_checkpoint_file(checkpoint, varsToIgnore)

with tf.contrib.slim.arg_scope(mobileNet_v3.training_scope(is_training=True)):
    logits, endpoints = mobileNet_v3.mobilenet(input_images, outputRegressionCount)

#getting the variables except for the last layer
#print(endpoints)
vars = []
for name in varList:
    vars = vars + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

saver = tf.train.Saver(vars)

#add optimizers to mobilenet
trainStep, mseLoss, maeLoss = constructTrainingParams(input_images, logits, labels)
summ_loss = tf.summary.scalar("mse_loss", mseLoss)
summ_mae_loss = tf.summary.scalar("mae_loss", maeLoss)

numberOfRunsPerBatch = len(image_files) //BATCH_SIZE


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #restore mobilenet
    saver.restore(sess, checkpoint)
    # tf.train.write_graph(sess.graph.as_graph_def(), '.', 'trainModel.pbtxt', as_text=True)

    # print(train_files)
    #create tensorboard summary file
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MobilenetV2")
    #create the saver for later
    train_saver = tf.train.Saver(vars)
    step = 0

    for e in range(EPOCHS):
        tot_loss = 0
        sess.run(iter.initializer,
                 feed_dict={filenames_placeholder: image_files, labels_inputPlaceholder: train_groundTruth,
                            batch_size: BATCH_SIZE})
        valStep = step
        for r in range(numberOfRunsPerBatch):
            _, loss_val, tb_loss_mae,tb_loss = sess.run([trainStep, mseLoss, summ_mae_loss, summ_loss])
            tot_loss += loss_val
            print("epoch: ", e, "batch num: ", r)
            train_writer.add_summary(tb_loss, step)
            train_writer.add_summary(tb_loss_mae, step)

            step += 1

        valStepStartingPoint = valStep
        for r in range(int(numberOfRunsPerBatch * percentValidation)):
            sess.run(iter.initializer, feed_dict={filenames_placeholder: validation_files,
                                                  labels_inputPlaceholder: validation_groundTruth,
                                                  batch_size: BATCH_SIZE})
            val_loss, val_sum_mae_loss = sess.run([summ_loss, summ_mae_loss])
            print('Validation step')
            validation_writer.add_summary(val_loss, valStep)
            validation_writer.add_summary(val_sum_mae_loss, valStep)
            valStep += int((step - valStepStartingPoint) / (numberOfRunsPerBatch * percentValidation))

        print("Iter: {}, Loss: {:.4f}".format(e, tot_loss / numberOfRunsPerBatch))
        tf.train.write_graph(sess.graph.as_graph_def(), '.', rootFolder + 'trainModel.pbtxt', as_text=True)
        train_saver.save(sess, CHECKPOINT_NAME)
        print("Saved checkpoint as: ", CHECKPOINT_NAME)

#create a new graph for exporting
#this graph doesn't contain any of the optimizers
# with tf.Graph().as_default() as eval_graph:
#     resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
#     with tf.contrib.slim.arg_scope(mobileNet_v3.training_scope(is_training=False)):
#         logits, endpoints = mobileNet_v3.mobilenet(resized_input_tensor, outputRegressionCount)
#
# with tf.Session(graph=eval_graph) as eval_sess: #not sure if you need to specify since it's already default but i'm just doing it to be safe
#     tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)
#
#     output_graph_def = tf.graph_util.convert_variables_to_constants(
#         eval_sess, eval_graph.as_graph_def(), [FLAGS.final_tensor_name])










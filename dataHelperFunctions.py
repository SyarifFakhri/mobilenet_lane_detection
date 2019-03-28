import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import math
import cv2
def extractLabels(pictureFileNames):
	"""
	Takes the picture file names (array) and finds the corresponding labels.
	assumes that the labels are formatted with one line spacing in between them and it's a simple regression problem
	:param pictureFileNames: the image file names to find the labels of

	:return: The labels of the images in order
	"""
	temp_labels = []
	for name in pictureFileNames:
		splitFiles = os.path.splitext(name)
		name = splitFiles[0]
		# do_label_filter = True
		text_file = open(name + ".txt", "r")
		lines = text_file.read().split()
		temp_labels.append(lines)
		text_file.close()
	return temp_labels

def addDatasetPath(datasetPath, iterable):
	"""
	Appends a dataset path (str) to each element of an iterable. Technically the same thing as a map function.
	:param datasetPath: String you want to append at the beginning
	:param iterable: array of elements to be added
	:return: array with added elements
	"""
	for i in range(len(iterable)):
		iterable[i] = datasetPath + iterable[i]
	return iterable

def filterImages(array):
	"""
	Given a list of all the filenames in a path, this will take the list and return
	filenames that only end in jpg
	:param array: array of filenames in a path
	:return: filtered list of filenames with images only
	"""
	newArr = []
	for name in array:
		listName = os.path.splitext(name)
		extension = listName[-1]
		if extension == ".jpg":
			newArr.append(name)
	return newArr

def print_tensors_in_checkpoint_file(file_name, varsToIgnore):
	"""Gets the variable list with the excpetion of some variables
	Inputs:
		file_name
		varsToIgnore: The vars to ignore, can be a list or a string
	Outputs:
		varlist: the list of variables
	"""
	varlist=[]
	reader = pywrap_tensorflow.NewCheckpointReader(file_name)
	var_to_shape_map = reader.get_variable_to_shape_map()
	for key in sorted(var_to_shape_map):
		if key in varsToIgnore:
			print("skipping key:", key)
			continue
		varlist.append(key)
	return varlist

def prepare_file_system(summaries_dir):
	"""
	deletes the summaries dir, then adds the summaries directory.
	:param summaries_dir:
	:return: None
	"""
	if tf.gfile.Exists(summaries_dir):
		tf.gfile.DeleteRecursively(summaries_dir)
	tf.gfile.MakeDirs(summaries_dir)
	return

def drawOneLane(_image, _a, _b, _c, color):
	colorDict = {"blue":(255,0,0), "green":(0,255,0), "yellow":(0,255,255), "red":(0,0,255)}
	height_hat = 1640
	width_hat = 590
	for i in range(height_hat):
		a = float(_a)
		b = float(_b)
		c = float(_c) - i

		if a == 0:
			a = 0.0001

		if (b ** 2 - (4 * a * c)) < 0:
			print("unsolvable")
			continue

		d = math.sqrt((b ** 2) - (4 * a * c))
		x = (-b - d) // (2 * a)
		# print('det: ', -b - d, 'x_hat:', x, ',y_hat:', i)
		cv2.circle(_image, (i, int(x)), 1, colorDict[color], thickness=-1, lineType=8, shift=0)
		#cv2.circle(_image, (int(x), i), 1, colorDict[color], thickness=-1, lineType=8, shift=0)
		x = (-b + d) // (2 * a)
		cv2.circle(_image, (i, int(x)), 1, colorDict[color], thickness=-1, lineType=8, shift=0)
		#cv2.circle(_image, (int(x), i), 1, colorDict[color], thickness=-1, lineType=8, shift=0)
	return _image

def drawOneLaneFromPoints(_image, pointsX, pointsY, color):
	colorDict = {"blue": (255, 0, 0), "green": (0, 255, 0), "yellow": (0, 255, 255), "red": (0, 0, 255)}
	assert len(pointsX) == len(pointsY), "number of pointsX and pointsY should be the same" + len(pointsX) + len(pointsY)

	for i in range(len(pointsX)):
		cv2.circle(_image, (int(pointsX[i]), int(pointsY[i])), 3, colorDict[color], -1)

	return _image





"""
WARNING: this code is destructive. It overides the dataset when it standardizes, so make sure
you keep a backup of the raw dataset before you run.
This code actually does a few things.
1. It normalizes the data set to have 0 mean and unit std deviation
2. It removes all unnescessary labels and only puts labels relevant to regression
3. It removes any data that doesn't contain the center lane

This data extracts the center lane from the CULane dataset
"""

import os
import random
from tensorflow import gfile
from dataHelperFunctions import filterImages, addDatasetPath, extractLabels
from statistics import mean, stdev
import create_validation_file


def normalize(data, mean, stdev):
    return str((data - mean) / stdev)

# dataset_path = "./filtered-filtered-dataset-twoLane-Normalized - validation incorrect/"
dataset_path = "./augment_all_v2/"
#reset the directory - If you wanna use this first move the pictures back because it doesn't do that automatically

#get all the images inside the directory
files = os.listdir(dataset_path)
image_files = filterImages(files)
image_files = addDatasetPath(dataset_path, image_files)
labels = extractLabels(image_files)

#I apologize to anyone that has to maintain this code, but it wasn't mean to be extendable, just one off and fast
#if you want to use it for a different dataset and with more coefficients, consider refactoring

#this is for normalization later
#divide by the max
aCoeffArray = []
bCoeffArray = []
cCoeffArray = []

aCoeffArrayLane2 = []
bCoeffArrayLane2 = []
cCoeffArrayLane2 = []

for line in labels:
    aCoeff = float(line[7])
    bCoeff = float(line[8])
    cCoeff = float(line[9])

    aCoeffArray.append(aCoeff)
    bCoeffArray.append(bCoeff)
    cCoeffArray.append(cCoeff)

    aCoeff = float(line[10])
    bCoeff = float(line[11])
    cCoeff = float(line[12])

    aCoeffArrayLane2.append(aCoeff)
    bCoeffArrayLane2.append(bCoeff)
    cCoeffArrayLane2.append(cCoeff)



#normalize by subtracting the mean and dividing by the standard deviation
#first find the mean and std deviation of each
#find the mean
aCoeffMean = mean(aCoeffArray)
bCoeffMean = mean(bCoeffArray)
cCoeffMean = mean(cCoeffArray)

aCoeffMeanLane2 = mean(aCoeffArrayLane2)
bCoeffMeanLane2 = mean(bCoeffArrayLane2)
cCoeffMeanLane2 = mean(cCoeffArrayLane2)

#find the stddev
aCoeffStdev = stdev(aCoeffArray)
bCoeffStdev = stdev(bCoeffArray)
cCoeffStdev = stdev(cCoeffArray)

aCoeffStdevLane2 = stdev(aCoeffArrayLane2)
bCoeffStdevLane2 = stdev(bCoeffArrayLane2)
cCoeffStdevLane2 = stdev(cCoeffArrayLane2)

for name in image_files:
    # check the labels
    name_split = name.split("/")
    extension_split = os.path.splitext(name_split[-1])
    text_file_src_name = dataset_path + extension_split[0] + ".txt"
    #move images
    text_file = open(text_file_src_name, "r")
    lines = text_file.read().split()
    text_file.close()

    #if theres no lanes on either left or right
    if int(lines[1]) == 0 or int(lines[2]) == 0:
        print("removed file with lines: ", lines, "name: ", name)
        os.remove(name) #remove the image
        os.remove(text_file_src_name) #remove the txt

    else:
        print(lines)
        #note that the label is formatted in this manner:
        #leftMostExistence...rightMostExistence (0-4) a b c (first lane - 7-9) ... a b c (last lane)
        #for first lane
        lines[7] = normalize(float(lines[7]), aCoeffMean, aCoeffStdev)
        lines[8] = normalize(float(lines[8]), bCoeffMean, bCoeffStdev)
        lines[9] = normalize(float(lines[9]), cCoeffMean, cCoeffStdev)
        #for second lane
        lines[10] = normalize(float(lines[10]), aCoeffMeanLane2, aCoeffStdevLane2)
        lines[11] = normalize(float(lines[11]), bCoeffMeanLane2, bCoeffStdevLane2)
        lines[12] = normalize(float(lines[12]), cCoeffMeanLane2, cCoeffStdevLane2)
        print(" ".join(lines[7:13]))
        print("---------------------")

        # #delete all contents within the file
        open(text_file_src_name, "w").close()

        # #write the new line data
        text_file = open(text_file_src_name, "w")
        text_file.write(" ".join(lines[7:13]))
        text_file.close()

text_to_write = str(aCoeffMean) + " " + \
                str(aCoeffStdev) + " " + \
                str(bCoeffMean) + " " + \
                str(bCoeffStdev) + " " + \
                str(cCoeffMean) + " " + \
                str(cCoeffStdev) + " " + \
                str(aCoeffMeanLane2) + " " + \
                str(aCoeffStdevLane2) + " " + \
                str(bCoeffMeanLane2) + " " + \
                str(bCoeffStdevLane2) + " " + \
                str(cCoeffMeanLane2) + " " + \
                str(cCoeffStdevLane2)

print(text_to_write)

"""
Normalization params contains the mean and std deviation of each coefficient
TODO - make it so that it is A_mean, A_stdev, B_mean, B_stdev...etc
"""
text_file = open(dataset_path + "NormalizationParams.txt", "w")
text_file.write(text_to_write)

#when done create the validation file
create_validation_file.create_validation_file(dataset_path)



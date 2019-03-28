import os
import statistics
import shutil

from dataHelperFunctions import filterImages, addDatasetPath, extractLabels

def normalize(data, mean, stdev):
    return str((data - mean) / stdev)

dataset_path = "D:/cuLane/culane_preprocessing/converted_dataset_percentage_augmented/" #don't forget the slash at the end!

new_dataset_path = "D:/LaneDetectionV2/d_aug_two_lanes_percentage_dataset/"

print("reading data...")
files = os.listdir(dataset_path)
image_files = filterImages(files)
image_files = addDatasetPath(dataset_path, image_files) #merging the full path with the image file names in order to copy them properly later
labels = extractLabels(image_files)

print("read all data...")

#calculate the mean of each
meansOfAllData = []
stdevOfAllData = []
for i in range(len(labels[0])): #assuming that all the length of data will be constant
    print("calculating mean and stddev at index: ", i)
    allDataAcrossSingleIndex = []
    for j in range(len(labels)):
        allDataAcrossSingleIndex.append(float(labels[j][i]))
    mean = statistics.mean(allDataAcrossSingleIndex)
    std = statistics.stdev(allDataAcrossSingleIndex)
    meansOfAllData.append(mean)
    stdevOfAllData.append(std)

print("mean: ", meansOfAllData)
print("std dev: ", stdevOfAllData)

#save the normalized params
text_file = open(new_dataset_path + "NormalizationParams.txt", "w")

for i in range(len(meansOfAllData)):
    #mean followed by stddev
    text_file.write(str(meansOfAllData[i]) + " ")
    text_file.write(str(stdevOfAllData[i]) + " ")

text_file.close()


print("saved normalization params to NormalizationParams.txt...")

for name in image_files:
    # get the src file name
    name_split = name.split("/")
    extension_split = os.path.splitext(name_split[-1])
    text_file_src_name = dataset_path + extension_split[0] + ".txt"

    #copy the image over
    shutil.copy(dataset_path + name_split[-1], new_dataset_path + name_split[-1])

    #read the text file
    text_file = open(text_file_src_name, "r")
    lines = text_file.read().split()
    text_file.close()

    new_text_file = open(new_dataset_path + extension_split[0] + ".txt", "w")

    for i in range(len(lines)):
        dataPoint = lines[i]
        meanOfDataPoint = meansOfAllData[i]
        stdDevOfDataPoint = stdevOfAllData[i]
        normalizedDataPoint = normalize(float(dataPoint), meanOfDataPoint, stdDevOfDataPoint)
        new_text_file.write(normalizedDataPoint + " ")
    new_text_file.close()

    print("wrote out ", new_text_file)












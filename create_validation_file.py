import os
import random
from tensorflow import gfile
from dataHelperFunctions import addDatasetPath, filterImages

def create_validation_file(dataset_path):
    # dataset_path = "./filtered-filtered-dataset-twoLane-Normalized/"
    validation_percent = 0.1
    validation_directory = "Validation"

    #reset the directory - If you wanna use this first move the pictures back because it doesn't do that automatically
    if not gfile.Exists(dataset_path + validation_directory):
        # gfile.DeleteRecursively(dataset_path + validation_directory)
        gfile.MakeDirs(dataset_path + validation_directory)

    #get all the images inside the directory
    files = os.listdir(dataset_path)
    image_files = filterImages(files)
    image_files = addDatasetPath(dataset_path, image_files)

    #split it into validation files
    validation_amount = int(validation_percent * len(image_files))

    #this will randomly pop an x amount of labels
    validation_files = [image_files.pop(random.randrange(len(image_files))) for _ in range(validation_amount)]

    #move the validation files to a new folder
    for name in validation_files:
        #move images
        name_split = name.split("/")
        new_dest = dataset_path + validation_directory + "/" + name_split[-1]
        os.rename(name, new_dest)
        print("moved: ", name, "to", new_dest)

        #move text files
        extension_split = os.path.splitext(name_split[-1])
        text_file_src_name = dataset_path + extension_split[0] + ".txt"
        new_txt_dest = dataset_path + validation_directory + "/" + extension_split[0] + ".txt"
        os.rename(text_file_src_name, new_txt_dest)
        print("moved: ", text_file_src_name, "to", new_txt_dest)
        print("---------------------")

# create_validation_file("./Full dataset - normalized/")
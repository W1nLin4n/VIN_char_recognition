import csv
import os
import shutil
import string
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def get_label_mapping(map_path):
    lbl_map_f = open(map_path)
    lbl_map = []
    for line in lbl_map_f:
        nums = [int(x) for x in line.split()]
        lbl_map.insert(nums[0], nums[1])
    return np.array(lbl_map)


def load_data(img_path, lbl_path):
    img = open(img_path, "rb")
    lbl = open(lbl_path, "rb")
    img.read(4)
    samples = int.from_bytes(img.read(4))
    height = int.from_bytes(img.read(4))
    width = int.from_bytes(img.read(4))
    lbl.read(8)
    label_data = np.zeros(samples, dtype="int32")
    image_data = np.zeros((samples, width, height), dtype="int32")
    for i in range(samples):
        if i % 5000 == 0:
            print("Samples loaded: {}".format(i))
        label_data[i] = int.from_bytes(lbl.read(1))
        for j in range(int(width * height)):
            image_data[i][j // height][j % height] = int.from_bytes(img.read(1))
    image_data = np.transpose(image_data, (0, 2, 1))
    return image_data, label_data


def is_character_needed(c):
    return (c in string.digits) or ((c in string.ascii_uppercase) and (c not in ['I', 'O', 'Q']))


def remove_unnecessary(img_data, lbl_data, lbl_map):
    old_to_new_map = []
    new_lbl_map = []
    new_index = 0
    for i in range(lbl_map.shape[0]):
        if is_character_needed(chr(lbl_map[i])):
            old_to_new_map.insert(i, new_index)
            new_lbl_map.insert(new_index, lbl_map[i])
            new_index += 1
        else:
            old_to_new_map.insert(i, -1)

    old_to_new_map = np.array(old_to_new_map, dtype="int32")
    new_lbl_map = np.array(new_lbl_map, dtype="int32")

    new_img_data = []
    new_lbl_data = []
    for i in range(lbl_data.shape[0]):
        new_lbl = old_to_new_map[lbl_data[i]]
        if new_lbl != -1:
            new_img_data.append(img_data[i])
            new_lbl_data.append(new_lbl)

    new_img_data = np.array(new_img_data, dtype="int32")
    new_lbl_data = np.array(new_lbl_data, dtype="int32")

    return new_img_data, new_lbl_data, new_lbl_map


def save_dataset(img_data, lbl_data, lbl_map, data_folder, data_lbl_file):
    shutil.rmtree(data_folder, ignore_errors=True)
    os.mkdir(data_folder)
    data_lbl_file = open(data_lbl_file, "w")
    writer = csv.writer(data_lbl_file)
    writer.writerow(['file', 'label'])

    for i in range(img_data.shape[0]):
        if i % 5000 == 0:
            print("Samples saved: {}".format(i))
        Image.fromarray(img_data[i]).convert("RGB") \
            .save(data_folder.joinpath("image{}.png".format(i)))
        writer.writerow(["image{}.png".format(i), chr(lbl_map[lbl_data[i]])])
    data_lbl_file.close()


def main():
    # Constants
    data_root = pathlib.Path(__file__).parent.absolute()
    data_raw_folder = data_root.joinpath("emnist_raw")
    data_modified_folder = data_root.joinpath("emnist_modified")

    train_img_raw_file = data_raw_folder.joinpath("emnist-train-images")
    test_img_raw_file = data_raw_folder.joinpath("emnist-test-images")
    train_lbl_raw_file = data_raw_folder.joinpath("emnist-train-labels")
    test_lbl_raw_file = data_raw_folder.joinpath("emnist-test-labels")
    raw_map_file = data_raw_folder.joinpath("emnist-mapping.txt")
    modified_lbl_file = data_modified_folder.joinpath("emnist_modified.csv")

    # Getting raw dataset info
    lbl_map = get_label_mapping(raw_map_file)
    print("Loading train set")
    train_img, train_lbl = load_data(train_img_raw_file, train_lbl_raw_file)
    print("Train set loaded")
    print("Loading test set")
    test_img, test_lbl = load_data(test_img_raw_file, test_lbl_raw_file)
    print("Test set loaded")

    # Merging train and test sets to make custom split
    img_data = np.concatenate((train_img, test_img))
    lbl_data = np.concatenate((train_lbl, test_lbl))

    # Removing unnecessary data
    img_data, lbl_data, lbl_map = remove_unnecessary(img_data, lbl_data, lbl_map)

    # Saving final dataset
    print("Saving final dataset")
    save_dataset(img_data, lbl_data, lbl_map, data_modified_folder, modified_lbl_file)
    print("Final dataset saved")


if __name__ == "__main__":
    main()

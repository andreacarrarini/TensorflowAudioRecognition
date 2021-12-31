import tensorflow as tf
import os
import pathlib

import numpy as np
import inspect

from tensorflow.keras import layers
from tensorflow.keras import models

import csv
import numpy as np
import librosa
import librosa.display

import sys


# This function opens the csv file
def open_csv_file(file_path):
    open_file = open(file_path, 'r')
    return open_file


# This function closes the csv file
def close_csv_file(open_file):
    open_file.close()


# The problem with this function is that in the example
# the audio files were mono channel and so it's different
def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return audio


# Finds the last / and slices the file path
def get_file_name_from_path(file_path):
    parts = tf.strings.split(input=file_path, sep='/')

    # file_name = file_path[file_path.rfind('/')+1:]
    return parts[-1]


def csv_file_to_list(csv_file):
    rows = []
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        rows.append(row)
    return rows


# Returns: -1 if a < b; 0 if a = b; 1 if a > b:
def compare_string_tensors(a, b):
    min_size_elem = a
    min_size = len(a)
    if len(b) < len(a):
        min_size_elem = b
        min_size = len(b)

    i = 0
    while i < min_size:
        print("a[i] is: " + str(a[i]))
        print("b[i] is: " + str(b[i]))
        # TODO CONTINUE HERE
        if a[i] < b[i]:
            return -1
        elif a[i] > b[i]:
            return 1
        elif a[i] == b[i]:
            i += 1
            if i == min_size:
                # if they're the same until the last char of the shortest one
                if len(a) == len(b):
                    return 0

                elif len(a) < len(b):
                    return -1

                elif len(a) > len(b):
                    return 1


# We need the label of the file and we can get it from the csv file
def get_label(csv_list, file_name):
    low = 0
    high = len(csv_list) - 1
    mid = 0

    step = 0

    b = tf.convert_to_tensor(file_name, dtype=tf.string)
    file_path_parts = tf.strings.split(b, sep='\\')
    b = tf.slice(file_path_parts, begin=[9], size=[1])

    while low <= high:
        print("Step: " + str(step))
        step+=1
        print("Low is: " + str(low))
        print("High is: " + str(high))

        mid = (high + low) // 2
        print("Mid is: " + str(mid))

        # TODO: Error is here, b (file_name) contains the whole path to the file and not only the file name
        a = tf.convert_to_tensor(csv_list[mid][0], dtype=tf.string)


        print("a is: " + str(a))
        print("b is: " + str(b))

        a = tf.io.decode_raw(a, tf.uint8)
        b = tf.io.decode_raw(b, tf.uint8)

        a = a.numpy()
        b = b.numpy()

        # TODO: Continue here and compare each character (represented by an int)
        # to simulate a comparison between 2 strings which cannot be done in tf between tensors od different shapes

        # If x is greater, ignore left half
        # if a < b:
        if compare_string_tensors(a, b) == -1:
            low = mid + 1
            print("smaller")
            continue

        # If x is smaller, ignore right half
        # elif a > b:
        elif compare_string_tensors(a, b) == 1:
            high = mid - 1
            print("greater")
            continue

        # means x is present at mid
        elif compare_string_tensors(a, b) == 0:
            print("Label found: " + csv_list[mid][7])
            return csv_list[mid][7]

        else:
            print("File: " + file_name + " is not present.")
            return -1

# Function that puts the 2 above together
# def get_waveform_and_label(file_path):
#   CSV_PATH = '/content/drive/MyDrive/MSA/UrbanSound8K/metadata/UrbanSound8K.csv'
#   open_metadata_file = open_csv_file(CSV_PATH)

#   file_name = get_file_name_from_path(file_path)

#   label = get_label(open_metadata_file, file_name)
#   audio_binary = tf.io.read_file(file_path)
#   waveform = decode_audio(audio_binary)

#   close_csv_file(open_metadata_file)

#   return waveform, label

# def get_waveform_and_label_2(file_path):
#     file_name = get_file_name_from_path(file_path)
#
#     label = get_label(csv_list, file_name)
#     audio_binary = tf.io.read_file(file_path)
#     waveform = decode_audio(audio_binary)
#
#     return waveform, label


def prepare_waveform_dataset(files_dataset, csv_list):
    waveform_label_dataset = []

    for file_path in files_dataset:
        waveform_label_couple = []

        file_name = get_file_name_from_path(file_path)
        print("File name is: " + file_name)

        label = get_label(csv_list, file_name)
        print("Label is: " + label)

        # audio_binary = tf.io.read_file(file_path)
        # waveform = decode_audio(audio_binary)

        sound_file, sampling_rate = librosa.load(str(file_path), sr=None)

        waveform_label_couple.append((sound_file, label))

        waveform_label_dataset.append(waveform_label_couple)

    return waveform_label_dataset

#sess = tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.__version__)

    # Set the seed value for experiment reproducibility.
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Get the data

    DATASET_PATH = '/Users/drugh/Documents/PycharmProjects/MSA_Project/UrbanSound8K'

    DATASET_PATH = os.path.join('C:',os.sep,'Users','drugh','Documents','PycharmProjects','MSA_Project','UrbanSound8K')
    print("DATASET_PATH is : " + DATASET_PATH)

    data_dir = pathlib.Path(DATASET_PATH)
    if data_dir.exists():
        print(data_dir)
    if not data_dir.exists():
        print("Data_dir doesn't exist!")
        # print(local_file_path)

    # inspect.getsourcefile(tf)

    # commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    # commands = commands[commands != 'README.md']
    # print('Commands:', commands)

    # train_set = tf.io.gfile.glob(str(data_dir) + '/audio/fold1/*.wav')
    train_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio','fold1','*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold2', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold3', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold4', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold6', '*.wav'))

    # train_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold2/*.wav')
    # train_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold3/*.wav')
    # train_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold4/*.wav')
    # train_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold6/*.wav')

    # for elem in train_set:
    #     print(elem)

    # test_set = tf.io.gfile.glob(str(data_dir) + '/audio/fold5/*.wav')
    test_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold5', '*.wav'))
    test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold7', '*.wav'))
    test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold8', '*.wav'))
    test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold9', '*.wav'))
    test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold10', '*.wav'))

    # test_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold7/*.wav')
    # test_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold8/*.wav')
    # test_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold9/*.wav')
    # test_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold10/*.wav')

    # test_file = tf.io.read_file(DATASET_PATH + '/audio/fold1/7061-6-0-0.wav')
    test_file = tf.io.read_file(os.path.join(DATASET_PATH,'audio','fold1','7061-6-0-0.wav'))

    test_audio, _ = tf.audio.decode_wav(contents=test_file)
    test_audio.shape

    AUTOTUNE = tf.data.AUTOTUNE

    # TODO: find a way to make this not add an additional "/" after every directory or remove them from evey element
    files_ds = tf.data.Dataset.from_tensor_slices(train_set)

    # for elem in files_ds:
    #     print(elem)

    # for elem in files_ds:
    #     print(elem)

    # CSV_PATH = '/content/drive/MyDrive/MSA/UrbanSound8K/metadata/UrbanSound8K.csv'
    # open_metadata_file = open_csv_file(csv_filepath)

    # pippo = '/content/drive/MyDrive/MSA/UrbanSound8K/metadata/UrbanSound8K.csv'
    # parts = tf.strings.split(input=pippo,sep='/')
    # print(parts[-1])

    CSV_PATH = '/Users/drugh/Documents/PycharmProjects/MSA_Project/UrbanSound8K/metadata/UrbanSound8K.csv'
    CSV_PATH = os.path.join('C:',os.sep,'Users','drugh','Documents','PycharmProjects',
                            'MSA_Project','UrbanSound8K','metadata','UrbanSound8K.csv')

    global OPEN_FILE
    OPEN_FILE = open_csv_file(CSV_PATH)

    # It's a list in which each element is a row of the csv file (list of columns)
    global csv_list
    csv_list = csv_file_to_list(OPEN_FILE)

    close_csv_file(OPEN_FILE)

    # Preparing the data structure: list[(waveform, label), ...]
    waveform_label_structure = prepare_waveform_dataset(files_ds, csv_list)

    # waveform_ds = files_ds.map(map_func=get_waveform_and_label_2 , num_parallel_calls=AUTOTUNE)

    # Transforming the data structure into a dataset
    waveform_label_dataset = tf.data.Dataset.from_tensor_slices(waveform_label_structure)

    rows = 3
    cols = 3
    n = rows * cols
    # fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

    for i, (audio, label) in enumerate(waveform_label_dataset.take(n)):
        r = i // cols
        c = i % cols
        # ax = axes[r][c]
        # ax.plot(audio.numpy())
        # ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        # label = label.numpy().decode('utf-8')
        # ax.set_title(label)

        librosa.display.waveplot(audio)

    # plt.show()

    for (audio, label) in enumerate(waveform_label_dataset.take(n)):
        print(audio, label)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
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
import matplotlib.pyplot as plt

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
# def decode_audio(audio_binary):
#     # Decode WAV-encoded audio files to `float32` tensors, normalized
#     # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
#     audio, _ = tf.audio.decode_wav(contents=audio_binary)
#     return audio


# Finds the last / and slices the file path
def get_file_name_from_path_TENSOR(file_path):
    parts = tf.strings.split(input=file_path, sep='\\')

    # file_name = file_path[file_path.rfind('/')+1:]
    return parts[-1]


def get_file_name_from_path(file_path):
    parts = file_path.split("\\")

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
        if a[i] < int(b[0][i]):
            return -1
        elif a[i] > int(b[0][i]):
            return 1
        elif a[i] == int(b[0][i]):
            i += 1
            if i == min_size:
                # if they're the same until the last char of the shortest one
                pippo = len(a)
                pluto = len(b)
                if len(a) == len(b[0]):
                    return 0

                elif len(a) < len(b[0]):
                    return -1

                elif len(a) > len(b[0]):
                    return 1


# We need the label of the file and we can get it from the csv file
def get_label(csv_list, file_name):
    low = 0
    high = len(csv_list) - 1
    mid = 0

    step = 0

    # b = tf.convert_to_tensor(file_name, dtype=tf.string)
    # file_path_parts = tf.strings.split(b, sep='\\')
    # b = tf.slice(file_path_parts, begin=[9], size=[1])
    # b = tf.io.decode_raw(b, tf.uint8)

    b = file_name

    while low <= high:
        mid = (high + low) // 2

        # a = tf.convert_to_tensor(csv_list[mid][0], dtype=tf.string)
        a = csv_list[mid][0]

        # a = tf.io.decode_raw(a, tf.uint8)
        #
        # a = a.numpy()
        # b = b.numpy()

        # to simulate a comparison between 2 strings which cannot be done in tf between tensors od different shapes

        # If x is greater, ignore left half
        # if a < b:
        # if compare_string_tensors(a, b) == -1:
        if a < b:
            low = mid + 1
            continue

        # If x is smaller, ignore right half
        # elif a > b:
        # elif compare_string_tensors(a, b) == 1:
        elif a > b:
            high = mid - 1
            continue

        # means x is present at mid
        # elif compare_string_tensors(a, b) == 0:
        elif a == b:
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
        # waveform_label_couple = []

        # ffff = tf.io.decode_raw(file_path, tf.uint8)
        # dddd = tf.strings.as_string(ffff)

        file_name = get_file_name_from_path(file_path)

        label = get_label(csv_list, file_name)

        # audio_binary = tf.io.read_file(file_path)
        # waveform = decode_audio(audio_binary)

        sound_file, sampling_rate = librosa.load(str(file_path), sr=None)

        # waveform_label_couple.append((sound_file, label))

        waveform_label_dataset.append((sound_file, label))

    return waveform_label_dataset


def data_generator(sound_arrays, labels):
    for i in range(len(sound_arrays)):
        audio_ragged_tensor = tf.ragged.constant(sound_arrays[i])
        label_ragged_tensor = tf.ragged.constant(labels[i])
        yield audio_ragged_tensor, label_ragged_tensor

def data_PORCODDIO(sound_arrays, labels):
    dataset_DIOCANE = []
    # for i in range(len(sound_arrays)):
    for i in range(3):
        audio_ragged_tensor = tf.ragged.constant(sound_arrays[i])
        label_ragged_tensor = tf.ragged.constant(labels[i])
        print(audio_ragged_tensor)
        print(label_ragged_tensor)
        dataset_DIOCANE.append((audio_ragged_tensor, label_ragged_tensor))
    return dataset_DIOCANE


# sess = tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.__version__)

    # Set the seed value for experiment reproducibility.
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Get the data

    DATASET_PATH = '/Users/drugh/Documents/PycharmProjects/MSA_Project/UrbanSound8K'

    DATASET_PATH = os.path.join('C:', os.sep, 'Users', 'drugh', 'Documents', 'PycharmProjects', 'MSA_Project',
                                'UrbanSound8K')
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
    train_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep, 'audio', 'fold1', '*.wav'))
    # train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold2', '*.wav'))
    # train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold3', '*.wav'))
    # train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold4', '*.wav'))
    # train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold6', '*.wav'))

    # train_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold2/*.wav')
    # train_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold3/*.wav')
    # train_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold4/*.wav')
    # train_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold6/*.wav')

    # for elem in train_set:
    #     print(elem)

    # test_set = tf.io.gfile.glob(str(data_dir) + '/audio/fold5/*.wav')
    test_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep, 'audio', 'fold5', '*.wav'))
    # test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold7', '*.wav'))
    # test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold8', '*.wav'))
    # test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold9', '*.wav'))
    # test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold10', '*.wav'))

    # test_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold7/*.wav')
    # test_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold8/*.wav')
    # test_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold9/*.wav')
    # test_set += tf.io.gfile.glob(str(data_dir) + '/audio/fold10/*.wav')

    # test_file = tf.io.read_file(DATASET_PATH + '/audio/fold1/7061-6-0-0.wav')
    test_file = tf.io.read_file(os.path.join(DATASET_PATH, 'audio', 'fold1', '7061-6-0-0.wav'))

    test_audio, _ = tf.audio.decode_wav(contents=test_file)
    test_audio.shape

    AUTOTUNE = tf.data.AUTOTUNE

    # Trying not to convert into a dataset
    # files_ds = tf.data.Dataset.from_tensor_slices(train_set)

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
    CSV_PATH = os.path.join('C:', os.sep, 'Users', 'drugh', 'Documents', 'PycharmProjects',
                            'MSA_Project', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')

    global OPEN_FILE
    OPEN_FILE = open_csv_file(CSV_PATH)

    # It's a list in which each element is a row of the csv file (list of columns)
    global csv_list
    csv_list = csv_file_to_list(OPEN_FILE)

    close_csv_file(OPEN_FILE)

    # Preparing the data structure: list[(waveform, label), ...]
    # waveform_label_structure = prepare_waveform_dataset(files_ds, csv_list)
    waveform_label_structure = prepare_waveform_dataset(train_set, csv_list)

    # waveform_ds = files_ds.map(map_func=get_waveform_and_label_2 , num_parallel_calls=AUTOTUNE)

    sound_arrays = []
    labels = []
    for elem in waveform_label_structure:
        sound_arrays.append(elem[0])
        label = []
        for char in elem[1]:
            label.append(ord(char))
        labels.append(np.array(label))
        # print(sound_arrays)
        # print(labels)

    # Transforming the data structure into a dataset
    # waveform_label_dataset = tf.data.Dataset.from_tensor_slices(sound_arrays, labels)
    # waveform_label_dataset = tf.data.Dataset.from_tensor_slices(sound_arrays)

    # waveform_label_dataset = tf.data.Dataset.from_generator(
    #     lambda: iter(zip(sound_arrays, labels)),
    #     output_types=(tf.float32, tf.int64),
    #     output_signature=(
    #         tf.TensorSpec(shape=(), dtype=tf.int32),
    #         tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32))
    # ).padded_batch(
    #     batch_size=32,
    #     padded_shapes=([None], ())
    # )

    # TODO: Understand if both sound_arrays and labels elems must be converted to tensors
    #       and also how to pad these tensors to the same size (this seems to be the issue here)
    # waveform_label_dataset = tf.data.Dataset.from_generator(
    #     data_generator,
    #     args=[sound_arrays, labels],
    #     output_signature=(
    #         tf.RaggedTensorSpec(shape=(None, 2), dtype=tf.float32)),
    # )

    waveform_label_dataset = data_PORCODDIO(sound_arrays, labels)
    print("Il Dataset è: " )
    print(waveform_label_dataset)

    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

    for i, (audio, label) in enumerate(waveform_label_dataset.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)

        librosa.display.waveplot(audio)

    plt.show()

    # for (audio, label) in enumerate(waveform_label_dataset.take(n)):
    #     print(audio, label)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

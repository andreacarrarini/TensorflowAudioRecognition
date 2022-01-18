import tensorflow as tf
import os
import pathlib

import numpy as np
import inspect

from librosa import display
from tensorflow.keras import layers
from tensorflow.keras import models

import csv
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

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


def prepare_waveform_dataset(files_dataset, csv_list, batch_size):
    waveform_label_dataset = []
    count = 0

    for file_path in files_dataset:

        if count == batch_size:
            break

        file_name = get_file_name_from_path(file_path)

        label = get_label(csv_list, file_name)

        sound_file, sampling_rate = librosa.load(str(file_path), sr=None)

        waveform_label_dataset.append((sound_file, label))

        count += 1

        # Removing the file from the file dataset, otherwise we will get the same batch_size elements each time
        files_dataset.remove(file_path)

    return waveform_label_dataset, files_dataset

def get_all_label_types(csv_list):
    label_types = []

    for elem in csv_list:

        # Skipping 1st row
        if csv_list.index(elem) == 0:
            continue

        if elem[7] in label_types:
            continue
        else:
            label_types.append(elem[7])

    return label_types


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

def data_PORCODDIO_CANE(waveform_label_structure):
    dataset_DIOCANE = []
    # for i in range(len(waveform_label_structure_TRAIN)):
    for i in range(3):
        audio_ragged_tensor = tf.ragged.constant(waveform_label_structure[i][0])
        # label_ragged_tensor = tf.ragged.constant(labels[i])
        # print(audio_ragged_tensor)
        # print(label_ragged_tensor)
        dataset_DIOCANE.append((audio_ragged_tensor, waveform_label_structure[i][1]))
    return dataset_DIOCANE

def get_max_length(sound_arrays):
    max_length = 0
    for elem in sound_arrays:
        if len(elem) > max_length:
            max_length = len(elem)
    return max_length

# max_in_dims is essentially the desired shape of the output.
# Note: this function will fail if you provide a shape that is strictly smaller than t in any dimension.
# in this case max_in_dims = [n, l] where n=1 (each tensor is a single array) and l=max_length of the tensors (384000)
def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    print(s)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)

def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
      waveform[0], frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`loading_batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_MFCC(waveform):
    # Extracts Mel-frequency cepstral coefficients
    # audio = librosa.core.istft(waveform.numpy())

    MFCC = librosa.feature.mfcc(y=waveform, sr=44100, n_mfcc = 50)
    return MFCC

def pad_up_to_SPECTROGRAM(t, max_in_dims, constant_values):
    s = tf.shape(t)
    tensor_shape_tuple = t.get_shape()
    tensor_shape_list = tensor_shape_tuple.as_list()
    paddings = [[0, max_in_dims - tensor_shape_list[0]]]
    new_t = tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)
    return new_t

def pad_up_to_MFCC(t, constant_values):
    paddings = [[0, 0], [0, 999-tf.shape(t)[1]]]
    new_t = tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)
    return new_t

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

def labels_to_IDs(label_tensors):
    label_types = []

    for elem in label_tensors:
        if elem in label_types:
            continue
        else:
            label_types.append(elem)

    return label_types

def build_dataset(waveform_label_structure):
    sound_arrays = []
    labels = []
    for elem in waveform_label_structure_TRAIN:
        sound_arrays.append(elem[0])
        labels.append(elem[1])

    # max_audio_length = get_max_length(sound_arrays)

    sound_tensors_array = []
    for elem in sound_arrays:
        t = tf.constant(elem)
        sound_tensor = pad_up_to_SPECTROGRAM(t, 384000, 0)
        sound_tensors_array.append(tf.reshape(sound_tensor, [1, 384000]))

    sound_tensors = tf.data.Dataset.from_tensor_slices(sound_tensors_array)
    label_tensors = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((sound_tensors, label_tensors))

    return dataset, label_tensors

def build_MFCC_dataset(waveform_label_structure, labels_types):
    MFCC_array = []
    labels_IDs = []
    for elem in waveform_label_structure:
        # sound_arrays.append(elem[0])
        # labels.append(elem[1])

        _mfcc = get_MFCC(elem[0])
        _label = labels_types.index(elem[1])

        t = tf.constant(_mfcc)
        # print(t.shape)
        sound_tensor = pad_up_to_MFCC(t, 0)

        MFCC_array.append(sound_tensor)
        labels_IDs.append(_label)


    mfcc_tensors = tf.data.Dataset.from_tensor_slices(MFCC_array)
    label_tensors = tf.data.Dataset.from_tensor_slices(labels_IDs)
    dataset = tf.data.Dataset.zip((mfcc_tensors, label_tensors))

    return dataset

def plot_dataset_examples(dataset):
    rows = 4
    cols = 4
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

    for i, (audio, label) in enumerate(dataset.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio[0].numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)

        librosa.display.waveplot(audio[0].numpy())

    plt.show()

    for waveform, label in dataset.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    # display.display(display.Audio(waveform, rate=16000))

    fig, axes = plt.subplots(2, figsize=(12, 8))

    timescale = np.arange(waveform.shape[1])
    axes[0].plot(timescale, waveform[0].numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 44100])

    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    plt.show()

def build_spectrograms_dataset(dataset, labels_types):
    spectrograms = []
    labels_IDs = []

    for waveform, label in dataset:
        _spectrogram = get_spectrogram(waveform)
        _label = labels_types.index(label)

        spectrograms.append(_spectrogram)
        labels_IDs.append(_label)

    spectrogram_tensors = tf.data.Dataset.from_tensor_slices(spectrograms)
    label_ID_tensors = tf.data.Dataset.from_tensor_slices(labels_IDs)
    return tf.data.Dataset.zip((spectrogram_tensors, label_ID_tensors))

# def build_MFCC_dataset(dataset, labels_types):
#     MFCC_array = []
#     labels_IDs = []
#
#     for waveform, label in dataset:
#         _mfcc = get_MFCC(waveform)
#         _label = labels_types.index(label)
#
#         MFCC_array.append(_mfcc)
#         labels_IDs.append(_label)
#
#     mfcc_tensors = tf.data.Dataset.from_tensor_slices(MFCC_array)
#     label_ID_tensors = tf.data.Dataset.from_tensor_slices(labels_IDs)
#     return tf.data.Dataset.zip((mfcc_tensors, label_ID_tensors))

def plot_spectrogram_dataset(dataset):
    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, (spectrogram, label_id) in enumerate(dataset.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(spectrogram.numpy(), ax)
        ax.set_title((labels_types[label_id.numpy()]))
        ax.axis('off')

    plt.show()

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

    # train_set = tf.io.gfile.glob(str(data_dir) + '/audio/fold1/*.wav')
    train_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep, 'audio', 'fold1', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold2', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold3', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold4', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold6', '*.wav'))

    test_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep, 'audio', 'fold5', '*.wav'))
    # test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold7', '*.wav'))
    # test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold8', '*.wav'))
    # test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold9', '*.wav'))
    # test_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold10', '*.wav'))

    validation_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold7', '*.wav'))

    CSV_PATH = '/Users/drugh/Documents/PycharmProjects/MSA_Project/UrbanSound8K/metadata/UrbanSound8K.csv'
    CSV_PATH = os.path.join('C:', os.sep, 'Users', 'drugh', 'Documents', 'PycharmProjects',
                            'MSA_Project', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')

    global OPEN_FILE
    OPEN_FILE = open_csv_file(CSV_PATH)

    # It's a list in which each element is a row of the csv file (list of columns)
    global csv_list
    csv_list = csv_file_to_list(OPEN_FILE)

    close_csv_file(OPEN_FILE)

    # IMPORTANT TO AVOID LOADING IN MEMORY ALL DATASET AT THE SAME TIME (currently trying batch_size * 4)
    # loading_batch_size = 256
    # loading_batch_size = 512
    loading_batch_size = 1024
    # loading_batch_size = 8192

    # Preparing the data structure: list[(waveform, label), ...]
    waveform_label_structure_TRAIN, train_set = prepare_waveform_dataset(train_set, csv_list, loading_batch_size)
    waveform_label_structure_TEST, test_set = prepare_waveform_dataset(test_set, csv_list, loading_batch_size)
    waveform_label_structure_VALIDATION, validation_set = prepare_waveform_dataset(validation_set, csv_list, loading_batch_size)

    TRAIN_dataset, label_tensors = build_dataset(waveform_label_structure_TRAIN)

    TEST_dataset, non_serve = build_dataset(waveform_label_structure_TEST)

    VALIDATIONS_dataset, non_serve = build_dataset(waveform_label_structure_VALIDATION)

    plot_dataset_examples(TRAIN_dataset)

    labels_types = get_all_label_types(csv_list)

    # TRAIN_dataset = build_spectrograms_dataset(TRAIN_dataset, labels_types)
    #
    # TEST_dataset = build_spectrograms_dataset(TEST_dataset, labels_types)
    #
    # VALIDATION_dataset = build_spectrograms_dataset(VALIDATIONS_dataset, labels_types)
    #
    # plot_spectrogram_dataset(TRAIN_dataset)

    TRAIN_dataset = build_MFCC_dataset(waveform_label_structure_TRAIN, labels_types)

    TEST_dataset = build_MFCC_dataset(waveform_label_structure_TEST, labels_types)

    VALIDATION_dataset = build_MFCC_dataset(waveform_label_structure_VALIDATION, labels_types)

    # TODO: I'm here
    batch_size = 64
    train_ds = TRAIN_dataset.batch(batch_size)
    val_ds = VALIDATION_dataset.batch(batch_size)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, _ in TRAIN_dataset.take(1):
        input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(labels_types)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=TRAIN_dataset.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    # Adam's (Neon Genesis Evangelion) model optimization

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # 10 Epochs example-training

    EPOCHS = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()

    # Evaluating the model performance
    test_audio = []
    test_labels = []

    for audio, label in TEST_dataset:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    # Confusion matrix
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=labels_types,
                yticklabels=labels_types,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

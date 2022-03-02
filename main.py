import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import functions as fn
import architectures as arch

def build_all_spectrograms_datasets(train_ds, val_ds, test_ds, labels_types, is_RNN):
    TRAIN_dataset = fn.build_spectrograms_dataset(train_ds, labels_types, is_RNN)
    VALIDATION_dataset = fn.build_spectrograms_dataset(val_ds, labels_types, is_RNN)
    TEST_dataset = fn.build_spectrograms_dataset(test_ds, labels_types, is_RNN)
    return TRAIN_dataset, VALIDATION_dataset, TEST_dataset

def build_all_MFCC_datasets(train_ds, val_ds, test_ds, labels_types, is_RNN):
    TRAIN_dataset = fn.build_MFCC_dataset(train_ds, labels_types, is_RNN)
    VALIDATION_dataset = fn.build_MFCC_dataset(val_ds, labels_types, is_RNN)
    TEST_dataset = fn.build_MFCC_dataset(test_ds, labels_types, is_RNN)
    return TRAIN_dataset, VALIDATION_dataset, TEST_dataset


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


    train_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep, 'audio', 'fold1', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold2', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold3', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold4', '*.wav'))
    train_set += tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold6', '*.wav'))

    # test_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep, 'audio', 'fold5', '*.wav'))
    # test_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold8', '*.wav'))
    test_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold10', '*.wav'))


    # validation_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold7', '*.wav'))
    validation_set = tf.io.gfile.glob(str(data_dir) + os.path.join(os.sep,'audio', 'fold9', '*.wav'))



    CSV_PATH = '/Users/drugh/Documents/PycharmProjects/MSA_Project/UrbanSound8K/metadata/UrbanSound8K.csv'
    CSV_PATH = os.path.join('C:', os.sep, 'Users', 'drugh', 'Documents', 'PycharmProjects',
                            'MSA_Project', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')

    global OPEN_FILE
    OPEN_FILE = fn.open_csv_file(CSV_PATH)

    # It's a list in which each element is a row of the csv file (list of columns)
    global csv_list
    csv_list = fn.csv_file_to_list(OPEN_FILE)

    print("Total number of files in the dataset: " + str(len(csv_list)))

    fn.close_csv_file(OPEN_FILE)

    # IMPORTANT TO AVOID LOADING IN MEMORY ALL DATASET AT THE SAME TIME (currently trying batch_size * 4)
    # loading_batch_size = 256
    loading_batch_size = 512
    # loading_batch_size = 1024
    # loading_batch_size = 2048
    # loading_batch_size = 4096
    # loading_batch_size = 8192
    # loading_batch_size = 99999

    # Preparing the data structure: list[(waveform, label), ...]
    waveform_label_structure_TRAIN, train_set = fn.prepare_waveform_dataset(train_set, csv_list, loading_batch_size)
    waveform_label_structure_TEST, test_set = fn.prepare_waveform_dataset(test_set, csv_list, loading_batch_size)
    waveform_label_structure_VALIDATION, validation_set = fn.prepare_waveform_dataset(validation_set, csv_list, loading_batch_size)

    labels_types = fn.get_all_label_types(csv_list)

    # Only for Spectrograms
    TRAIN_dataset, label_tensors = fn.build_dataset(waveform_label_structure_TRAIN)
    fn.plot_dataset_examples(TRAIN_dataset, False)

    TEST_dataset, non_serve = fn.build_dataset(waveform_label_structure_TEST)

    VALIDATION_dataset, non_serve = fn.build_dataset(waveform_label_structure_VALIDATION)

    # For CNNs/VGG16/ResNet50 - Spectrogram
    TRAIN_dataset, VALIDATION_dataset, TEST_dataset = build_all_spectrograms_datasets(
        TRAIN_dataset, VALIDATION_dataset, TEST_dataset, labels_types, False)
    fn.plot_spectrogram_dataset(TRAIN_dataset, False, labels_types)

    # For CNNs/VGG16/ResNet50 - MFCC
    # TRAIN_dataset, VALIDATION_dataset, TEST_dataset = build_all_MFCC_datasets(
    #         waveform_label_structure_TRAIN, waveform_label_structure_VALIDATION,
    #         waveform_label_structure_TEST, labels_types, False)

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

    model = arch.build_CNN(num_labels, train_ds, input_shape)

    # model = arch.build_CNN_2(num_labels, train_ds, input_shape)

    # model = arch.build_ResNet50(num_labels, train_ds, input_shape)

    # model = arch.build_VGG16(num_labels, train_ds, input_shape)

    model.build(input_shape)
    model.summary()

    # Adam's (Neon Genesis Evangelion) model optimization

    # CNN
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                           amsgrad=False, name='Adam'),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 50
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

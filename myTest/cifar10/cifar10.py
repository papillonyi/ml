import os
import sys

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.training.rmsprop import RMSPropOptimizer

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
INPUT_TENSOR_NAME = 'inputs_input'
BATCH_SIZE = 128
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)


def keras_model_fn(hyperparameters):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', name='inputs', input_shape=(HEIGHT, WIDTH, DEPTH)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    opt = RMSPropOptimizer(learning_rate=hyperparameters['learning_rate'], decay=hyperparameters['decay'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def serving_input_fn(hyperparameters):
    tensor = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    """Returns input function that would feed the model during training"""
    return _input(tf.estimator.ModeKeys.TRAIN,
                    batch_size=BATCH_SIZE, data_dir=training_dir)


def eval_input_fn(training_dir, hyperparameters):
    """Returns input function that would feed the model during evaluation"""
    return _input(tf.estimator.ModeKeys.EVAL,
                    batch_size=BATCH_SIZE, data_dir=training_dir)


def _input(mode, batch_size, data_dir):
    """Uses the tf.data input pipeline for CIFAR-10 dataset.
    Args:
        mode: Standard names for model modes (tf.estimators.ModeKeys).
        batch_size: The number of samples per batch of input requested.
    """
    dataset = _record_dataset(_filenames(mode, data_dir))

    # For training repeat forever.
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat()

    dataset = dataset.map(_dataset_parser)
    dataset.prefetch(2 * batch_size)

    # For training, preprocess the image and shuffle.
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.map(_train_preprocess_fn)
        dataset.prefetch(2 * batch_size)

        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Subtract off the mean and divide by the variance of the pixels.
    dataset = dataset.map(
        lambda image, label: (tf.image.per_image_standardization(image), label))
    dataset.prefetch(2 * batch_size)

    # Batch results by up to batch_size, and then fetch the tuple from the
    # iterator.
    iterator = dataset.batch(batch_size).make_one_shot_iterator()
    images, labels = iterator.get_next()

    # We must use the default input tensor name PREDICT_INPUTS
    print(labels)
    return {INPUT_TENSOR_NAME: images}, labels


def _record_dataset(filenames):
    """Returns an input pipeline Dataset from `filenames`."""
    record_bytes = HEIGHT * WIDTH * DEPTH + 1
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def _filenames(mode, data_dir):
    """Returns a list of filenames based on 'mode'."""
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    assert os.path.exists(data_dir), ('Run cifar10_download_and_extract.py first '
                                      'to download and extract the CIFAR-10 data.')

    if mode == tf.estimator.ModeKeys.TRAIN:
        return [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(1, NUM_DATA_BATCHES + 1)
        ]
    elif mode == tf.estimator.ModeKeys.EVAL:
        return [os.path.join(data_dir, 'test_batch.bin')]
    else:
        raise ValueError('Invalid mode: %s' % mode)

def _train_preprocess_fn(image, label):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image, label


def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
    # Every record consists of a label followed by the image, with a fixed number
    # of bytes for each.
    label_bytes = 1
    image_bytes = HEIGHT * WIDTH * DEPTH
    record_bytes = label_bytes + image_bytes

    # Convert from a string to a vector of uint8 that is record_bytes long.
    raw_record = tf.decode_raw(value, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32.
    label = tf.cast(raw_record[0], tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(raw_record[label_bytes:record_bytes],
                             [DEPTH, HEIGHT, WIDTH])

    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    return image, tf.one_hot(label, NUM_CLASSES)


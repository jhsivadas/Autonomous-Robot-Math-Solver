"""
Neural Network to classify digits using the MNIST dataset. We use a few data augmentation
processes to format the digits similar to the digits we obtain from the whiteboard.
NOTE: This script requires the Python Image Library (PIL) and Tensorflow.
"""
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image
from typing import Tuple


THRESHOLD = 0.2


def get_first_nonzero(vals: np.ndarray) -> int:
    idx = 0
    while idx < len(vals):
        if abs(vals[idx] - 1.0) < 1e-8:
            return idx
        idx += 1

    return -1


def get_last_nonzero(vals: np.ndarray) -> int:
    idx = len(vals) - 1
    while idx >= 0:
        if abs(vals[idx] - 1.0) < 1e-8:
            return idx
        idx -= 1

    return -1


def resize_images(X_train: np.ndarray) -> np.ndarray:
    result: List[np.ndarray] = []

    rand = np.random.RandomState(seed=321890)

    for img in X_train:
        x_max = np.max(img, axis=-1)  # [28]
        row_start, row_end = get_first_nonzero(x_max), get_last_nonzero(x_max)

        y_max = np.max(img, axis=0)  # [28]
        col_start, col_end = get_first_nonzero(y_max), get_last_nonzero(y_max)

        clipped = img[row_start:(row_end + 1), col_start:(col_end + 1)]

        pillow_img = Image.fromarray(clipped)
        resized = np.array(pillow_img.resize((28, 28), Image.BILINEAR))

        resized = (resized > THRESHOLD).astype(float)
        result.append(np.expand_dims(resized, axis=0))

    return np.vstack(result)


def make_model() -> keras.Model:
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28, 1), name='input'))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', name='conv0'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', name='conv1'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten(name='flatten'))
    model.add(keras.layers.Dense(64, activation='relu', name='dense_hidden'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(10, activation='softmax', name='output'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train.astype(float) / 255.0
    X_train = (X_train > THRESHOLD).astype(float)

    X_train = resize_images(X_train)
    X_train = np.expand_dims(X_train, axis=-1)

    model = make_model()

    val_split = 0.2
    epochs = 10

    model.fit(X_train, y_train, batch_size=16, epochs=epochs, shuffle=True, validation_split=val_split)
    model.save('mnist.h5')

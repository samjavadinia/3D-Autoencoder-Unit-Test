import os
import tensorflow as tf
import shutil
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    Conv3DTranspose,
    BatchNormalization,
)
from tensorflow.keras.callbacks import TensorBoard, Callback
import h5py
from datetime import datetime
import numpy as np
import pdb
import matplotlib.pyplot as plt
import random
from tqdm.keras import TqdmCallback


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def define_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)
    print("####Input Layer###", input_layer.shape)

    encoder = Conv3D(64, kernel_size=(3, 3, 3), activation="relu", padding="same")(
        input_layer
    )
    print("####Input Layer###", encoder.shape)

    encoder = BatchNormalization()(encoder)
    print("####Input Layer###", encoder.shape)

    encoder = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(encoder)
    print("####Input Layer###", encoder.shape)

    encoder = Conv3D(128, kernel_size=(3, 3, 3), activation="relu", padding="same")(
        encoder
    )

    encoder = BatchNormalization()(encoder)
    encoder = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(encoder)

    # Decoder
    decoder = Conv3DTranspose(
        128, kernel_size=(3, 3, 3), activation="relu", padding="same"
    )(encoder)
    decoder = BatchNormalization()(decoder)
    decoder = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(decoder)
    print("####output_layer###", decoder.shape)

    decoder = Conv3DTranspose(
        64, kernel_size=(3, 3, 3), activation="relu", padding="same"
    )(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(decoder)

    print("####output_layer###", decoder.shape)

    # Output layer with the same number of channels as input data
    output_layer = Conv3D(
        1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(decoder)
    output_layer = output_layer * 255
    print("####output_layer###", output_layer.shape)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(
        optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError()
    )  # Adjust loss function if needed

    # print('####autoencoder###',autoencoder)

    return autoencoder

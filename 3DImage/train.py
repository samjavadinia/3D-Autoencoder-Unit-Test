import argparse
import numpy as np
import yaml
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import define_autoencoder
from data_generation import gen_samples, save_single_sample
from Inference import Inference_sample_volume


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        return config


def main(config_path="config.yml"):

    config = load_config(config_path)

    print("Generating synthetic data...")

    input_shape = (*config["image_size"], 1)
    autoencoder = define_autoencoder(input_shape)

    dataset = tf.data.Dataset.from_generator(
        lambda: gen_samples(
            config["batch_size"], config["radius"], config["image_size"]
        ),
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *input_shape], [None, *input_shape]),
    )

    print("dataset", dataset)
    # Configure the dataset for performance
    # dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print("dataset", dataset)

    sample_volume = save_single_sample(dataset)

    steps_per_epoch = config["num_samples"] // config["batch_size"]

    print(f"steps_per_epoch{steps_per_epoch}")
    # Train the autoencoder using the dataset
    autoencoder.fit(
        dataset, steps_per_epoch=steps_per_epoch, epochs=100
    )  # Changed line

    autoencoder.save("autoencoder_model.h5")
    predicted_output = Inference_sample_volume(
        sample_volume=sample_volume, model=autoencoder
    )


if __name__ == "__main__":
    main()

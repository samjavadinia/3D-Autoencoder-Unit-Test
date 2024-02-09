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
    # config = load_config(config_path)

    # print("Generating synthetic data...")
    # generate_and_save_volumes(
    #     config["num_samples"],
    #     tuple(config["image_size"]),
    #     config["radius"],
    #     config["data_path"],
    # )

    # # Load and preprocess the generated data
    # volumes = load_data(config["data_path"])
    # volumes = preprocess_data(volumes)

    # # Split data into training and validation sets
    # train_volumes, val_volumes = train_test_split(
    #     volumes, test_size=config["validation_split"], random_state=0
    # )

    # # Convert the NumPy arrays into tf.data.Dataset objects
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_volumes, train_volumes))
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_volumes, val_volumes))

    # # Batch the datasets
    # train_dataset = train_dataset.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)
    # val_dataset = val_dataset.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)

    # # Define the model
    # input_shape = train_volumes.shape[
    #     1:
    # ]  # Assuming data is shaped as (samples, height, width, depth, channels)
    # print
    # autoencoder = define_autoencoder(input_shape)

    # # Training
    # print("Training the model...")
    # autoencoder.fit(train_dataset, epochs=config["epochs"], validation_data=val_dataset)

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

    # Configure the dataset for performance
    # dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    sample_volume = save_single_sample(dataset)

    steps_per_epoch = 1000 // config["batch_size"]
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

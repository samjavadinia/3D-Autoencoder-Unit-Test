import numpy as np
import random
import nibabel as nib


# def generate_3d_volume(image_size, radius, center_sphere):
#     # Create a 3D grid
#     x = np.linspace(0, 1, image_size[0])
#     y = np.linspace(0, 1, image_size[1])
#     z = np.linspace(0, 1, image_size[2])
#     x, y, z = np.meshgrid(x, y, z, indexing="ij")

#     # Sphere mask using the fixed radius and center position
#     sphere_mask = (
#         (x - center_sphere[0]) ** 2
#         + (y - center_sphere[1]) ** 2
#         + (z - center_sphere[2]) ** 2
#     ) < (radius / image_size[0]) ** 2
#     # Initialize volume with grey for background
#     volume = np.full(image_size, 255)  # Set entire volume to grey

#     # Set the sphere to white
#     volume[sphere_mask] = 128  # Make the sphere white

#     return volume


def generate_3d_volume(image_size, radius, center_sphere):
    # Create a 3D grid
    x = np.linspace(0, 1, image_size[0])
    y = np.linspace(0, 1, image_size[1])
    z = np.linspace(0, 1, image_size[2])
    x, y, z = np.meshgrid(
        x, y, z, indexing="ij"
    )  # Ensure correct indexing for the grid

    # Sphere mask using the fixed radius and center position
    sphere_mask = (
        (x - center_sphere[0]) ** 2
        + (y - center_sphere[1]) ** 2
        + (z - center_sphere[2]) ** 2
    ) < (radius / image_size[0]) ** 2

    # Initialize volume with grey for background
    volume = np.full(image_size, 255)  # Set entire volume to grey

    # Set the sphere to white
    volume[sphere_mask] = 128  # Make the sphere white

    return volume


def gen_samples(batch_size, radius, image_size):
    while True:
        vols = []
        for _ in range(batch_size):
            sphere_center = (
                random.uniform(radius / image_size[0], 1 - radius / image_size[0]),
                random.uniform(radius / image_size[1], 1 - radius / image_size[1]),
                random.uniform(radius / image_size[2], 1 - radius / image_size[2]),
            )
            vol = generate_3d_volume(image_size, radius, sphere_center)
            vol = np.expand_dims(vol, axis=-1)  # Add channel dimension
            vols.append(vol)
        yield np.array(vols), np.array(vols)


def generate_and_save_volumes(num_samples, image_size, radius, save_path):
    volumes = []
    for i in range(num_samples):
        sphere_center = (
            random.uniform(radius / image_size[0], 1 - radius / image_size[0]),
            random.uniform(radius / image_size[1], 1 - radius / image_size[1]),
            random.uniform(radius / image_size[2], 1 - radius / image_size[2]),
        )
        volume = generate_3d_volume(image_size, radius, sphere_center)
        volumes.append(volume)
    np.savez_compressed(save_path, volumes=np.array(volumes))


def load_data(data_path):
    with np.load(data_path) as data:
        volumes = data["volumes"]
    return volumes


def preprocess_data(volumes):
    volumes = np.expand_dims(volumes, axis=-1)  # Add channel dimension
    volumes = volumes.astype("float32") / np.max(volumes)  # Normalize to [0, 1]
    return volumes


def save_single_sample(dataset):
    # Assuming dataset is your tf.data.Dataset object from earlier
    for inputs, _ in dataset.take(1):  # Take one batch from the dataset
        sample_volume = inputs.numpy()[
            0
        ]  # Convert the first sample of the batch to a NumPy array

        sample_volume = np.squeeze(sample_volume)

        # Create a NIfTI image. Assuming no specific affine transformation, use an identity matrix.
        # Adjust the affine matrix if you have specific spatial orientation requirements.
        nifti_img = nib.Nifti1Image(sample_volume, affine=np.eye(4))

        # Save the NIfTI image to disk
        nib.save(nifti_img, "sample_volume.nii.gz")
    return sample_volume

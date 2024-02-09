import numpy as np
import random
import nibabel as nib


def Inference_sample_volume(sample_volume_path=None, sample_volume=None, model=None):
    # Assuming dataset is your tf.data.Dataset object from earlier
    if sample_volume_path != None:
        # Load the NIfTI file
        nifti_sample_path = sample_volume_path
        nifti_sample = nib.load(nifti_sample_path)
        # Get the data as a numpy array
        sample_volume = nifti_sample.get_fdata()
    elif sample_volume != None:

        sample_volume = sample_volume

    else:
        raise TypeError("Missing required argument 'name'")

    sample_volume = np.expand_dims(sample_volume, axis=0)  # Add batch dimension
    sample_volume = np.expand_dims(
        sample_volume, axis=-1
    )  # Add channel dimension if needed

    # Ensure the sample data type matches your model's expected input type, typically float32
    sample_volume = sample_volume.astype(np.float32)

    # Assuming autoencoder is your trained model
    predicted_output = model.predict(sample_volume)

    # Remove batch dimension and channel dimension if added before
    predicted_output = np.squeeze(predicted_output)

    # Create a NIfTI image for the output
    # Use the same affine transformation as the input for consistency
    output_nifti_img = nib.Nifti1Image(predicted_output, affine=nifti_sample.affine)

    # Save the output NIfTI image
    nib.save(output_nifti_img, "predicted_output.nii.gz")

    return predicted_output

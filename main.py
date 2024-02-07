import os
import numpy as np
import shutil
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# Remove existing logs directory
logs_directory = 'logs'
if os.path.exists(logs_directory):
    shutil.rmtree(logs_directory)
# Load 3D medical image data
def load_data(data_dir):
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    image_files = sorted(os.listdir(images_dir))
    label_files = sorted(os.listdir(labels_dir))

    X = []
    Y = []

    for img_file, lbl_file in zip(image_files, label_files):
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir, lbl_file)

        # Assuming you have a function to load 3D medical images and labels
        img = load_medical_image(img_path)
        # print(img.shape)
        lbl = load_medical_label(lbl_path)
        # print(img.shape)
        # exit(1)
        

        X.append(img)
        Y.append(lbl)
        
       
    return np.array(X), np.array(Y)

# Replace this with your actual data loading logic for medical images
# def load_medical_image(file_path):
#     # Replace this with your actual data loading logic for medical images
#     # This is just a placeholder
#     return np.random.rand(32, 64, 64, 1)

# # Replace this with your actual data loading logic for medical labels
# def load_medical_label(file_path):
#     # Replace this with your actual data loading logic for medical labels
#     # This is just a placeholder
#     return np.random.rand(32, 64, 64, 1)
def load_medical_image(file_path):
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)
    # Assuming you want to resize to (32, 64, 64) for consistency
    resized_image = resize_image(image_array, (32, 64, 64))
    return resized_image[..., np.newaxis]  # Add a channel dimension

def load_medical_label(file_path):
    label = sitk.ReadImage(file_path)
    label_array = sitk.GetArrayFromImage(label)
    # Assuming you want to resize to (32, 64, 64) for consistency
    resized_label = resize_image(label_array, (32, 64, 64))
    return resized_label[..., np.newaxis]  # Add a channel dimension

def resize_image(image, new_size):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resized_image = resampler.Execute(sitk.GetImageFromArray(image))
    return sitk.GetArrayFromImage(resized_image)

# Create a 3D autoencoder model
def create_3d_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((2, 2, 2), padding='same')(x)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')

    return autoencoder

# Set the path to your downloaded dataset
data_dir = "/home/samanehjavadinia/3dimage/Task04_Hippocampus"

# Load 3D medical image data
X, Y = load_data(data_dir)
import pdb
pdb.set_trace()
    

# Assuming your data is binary (0s and 1s), you may need to normalize it between 0 and 1
X = X.astype('float32') / np.max(X)
Y = Y.astype('float32') / np.max(Y)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the shape of the training, validation sets
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"Y_val shape: {Y_val.shape}")

# Create the 3D autoencoder model
input_shape = X_train.shape[1:]
autoencoder = create_3d_autoencoder(input_shape)

# Set up TensorBoard
log_dir = "./logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

def create_image_summary(epoch, images, name, max_outputs=4):
    # Create a summary writer
    with tf.summary.create_file_writer(log_dir).as_default():
        # Reshape the images to (batch_size * depth, height, width, channels)
        reshaped_images = tf.reshape(images, [-1] + list(images.shape[2:]))
        # Log images
        tf.summary.image(name, reshaped_images, max_outputs=max_outputs, step=epoch)


# Train the autoencoder
epochs = 10
batch_size = 32  # Adjust this based on your system's memory capacity

for epoch in range(epochs):
    autoencoder.fit(X_train, Y_train,
                    epochs=1,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_val, Y_val),
                    callbacks=[tensorboard_callback])
    
    # Visualize input and reconstructed images in TensorBoard
    reconstructed_images = autoencoder.predict(X_val[:4])  # Assuming you want to visualize the first 4 validation samples
    create_image_summary(epoch, X_val[:4], 'Input Images')
    create_image_summary(epoch, reconstructed_images, 'Reconstructed Images')

# Evaluate the autoencoder on the validation set
evaluation = autoencoder.evaluate(X_val, Y_val, batch_size=batch_size)
print(f"Validation Loss: {evaluation}")

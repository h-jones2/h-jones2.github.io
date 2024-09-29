import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Image dimensions
img_height = 64
img_width = 64
batch_size = 2  # Adjust batch size for small dataset

# Path to your dataset folder
dataset_folder = r'C:\Users\jrani\OneDrive\Desktop\dataset'

# Create ImageDataGenerator instance
train_datagen = ImageDataGenerator(rescale=1./255)  # No validation split for now

# Load training data
train_data = train_datagen.flow_from_directory(
    dataset_folder,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Check if train_data is loaded correctly
print(f"Found {train_data.samples} images belonging to {train_data.num_classes} classes.")

# Print out which pictures belong to the handwritten or printed class
class_labels = train_data.class_indices  # Get class labels
print("Class labels:", class_labels)

# Loop through the batches of images and print the filenames with their class
for i in range(len(train_data)):
    images, labels = train_data.next()  # Get the next batch of images and labels
    for j in range(len(images)):
        class_name = 'Handwritten' if labels[j] == 0 else 'Printed'
        print(f"Image {i * batch_size + j + 1}: Class = {class_name}")

# Build a simple CNN model
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
if train_data.samples > 0:  # Only fit if we have training data
    history = model.fit(train_data, epochs=10, batch_size=batch_size)
else:
    print("No training data found!")

# Save the model
model.save('handwritten_vs_printed_model.h5')

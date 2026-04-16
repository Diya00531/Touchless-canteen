import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Dataset folder
dataset_path = "dataset"

# Image augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

# Load dataset
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=4,
    class_mode="categorical"
)

print("Classes Found:", train_data.class_indices)

# CNN Model
model = Sequential()

model.add(Input(shape=(128,128,3)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(len(train_data.class_indices), activation='softmax'))

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining started...\n")

# Train
model.fit(
    train_data,
    epochs=10
)

# Save model
model.save("food_model.keras")

print("\nModel trained and saved as food_model.keras")
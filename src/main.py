import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Path for the model to save
model_path = 'pneumonia_vs_normal.h5'

# If the model exists, delete it
if os.path.exists(model_path):
    os.remove(model_path)
    print(f"Deleted existing model file: {model_path}")


def adjusted_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    elif epoch < 10:
        return lr * 0.2
    else:
        return lr * 0.02


# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generator for the validation set
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=64,  # Increased batch size
    class_mode='binary',
    shuffle=True
)

# Flow validation images in batches using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(150, 150),
    batch_size=64,  # Increased batch size to match training
    class_mode='binary',
    shuffle=True
)

# Building the model
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),  # Slightly lower dropout rate
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1, activation='sigmoid')
])


# Compiling the model with a lower learning rate
initial_learning_rate = 1e-4
model.compile(optimizer=Adam(learning_rate=initial_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate scheduling
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
lr_schedule = LearningRateScheduler(adjusted_scheduler)

# Fitting the model to the data
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    shuffle=True,  # Shuffle the data
    callbacks=[early_stopping, lr_schedule]
)

# Saving the model
model.save('my_model.keras')

# Plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_plot.png')

# Clear the current figure
plt.clf()

# Plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss_plot.png')

plt.clf()

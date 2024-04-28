from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from tensorflow.keras.metrics import Precision, Recall, AUC
import matplotlib.pyplot as plt
import os

model_path = 'efficientnet_model.h5'
if os.path.exists(model_path):
    os.remove(model_path)
    print(f"Deleted existing model file: {model_path}")


class MetricsCallback(Callback):
    def on_train_begin(self, logs=None):
        self.best_f1 = -1
        self.best_epoch = -1
        self.best_accuracy = -1
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        precision = logs['precision']
        recall = logs['recall']
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

        if f1_score > self.best_f1:
            self.best_f1 = f1_score
            self.best_epoch = epoch

        # Track best accuracy
        if logs['accuracy'] > self.best_accuracy:
            self.best_accuracy = logs['accuracy']

        # Track best loss
        if logs['loss'] < self.best_loss:
            self.best_loss = logs['loss']

        print(
            f"Epoch {epoch + 1}: F1 score = {f1_score:.4f}, Accuracy = {logs['accuracy']:.4f}, Loss = {logs['loss']:.4f}")

    def on_train_end(self, logs=None):
        print(f"Best F1 score {self.best_f1:.4f} was achieved at epoch {self.best_epoch + 1}")
        print(f"Best accuracy {self.best_accuracy:.4f} was achieved at epoch {self.best_epoch + 1}")
        print(f"Best loss {self.best_loss:.4f} was achieved at epoch {self.best_epoch + 1}")


def adjusted_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    elif epoch < 10:
        return lr * 0.2
    else:
        return lr * 0.02


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

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    shuffle=True
)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of EfficientNet
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# This is the new model
model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy',
              metrics=['accuracy', Precision(), Recall(), AUC(name='auc')])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(adjusted_scheduler)
metrics_callback = MetricsCallback()

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, lr_scheduler, metrics_callback]
)

# Save the model
model.save(model_path)


def plot_metric(history, metric, title, ylabel):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{metric}_plot.png')
    plt.clf()

plot_metric(history, 'accuracy', 'Model Accuracy', 'Accuracy')
plot_metric(history, 'loss', 'Model Loss', 'Loss')
plot_metric(history, 'auc', 'Model AUC', 'AUC')

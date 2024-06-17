from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from IPython.display import Image
import os

model_path = 'my_model_functional.keras'

if os.path.exists(model_path):
    os.remove(model_path)
    print(f"Deleted existing model file: {model_path}")


# Define the custom callback class
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

        if logs['accuracy'] > self.best_accuracy:
            self.best_accuracy = logs['accuracy']

        if logs['loss'] < self.best_loss:
            self.best_loss = logs['loss']

        print(
            f"Epoch {epoch + 1}: F1 score = {f1_score:.4f}, Accuracy = {logs['accuracy']:.4f}, Loss = {logs['loss']:.4f}")

    def on_train_end(self, logs=None):
        print(f"Best F1 score {self.best_f1:.4f} was achieved at epoch {self.best_epoch + 1}")
        print(f"Best accuracy {self.best_accuracy:.4f} was achieved at epoch {self.best_epoch + 1}")
        print(f"Best loss {self.best_loss:.4f} was achieved at epoch {self.best_epoch + 1}")


# Define the learning rate scheduler
def adjusted_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    elif epoch < 10:
        return lr * 0.2
    else:
        return lr * 0.02


# Data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

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

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    shuffle=False
)

# Model definition
inputs = Input(shape=(150, 150, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

initial_learning_rate = 1e-4
model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(), Recall(), AUC(name='auc')]
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
lr_schedule = LearningRateScheduler(adjusted_scheduler)
metrics_callback = MetricsCallback()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    shuffle=True,
    callbacks=[early_stopping, lr_schedule, metrics_callback]
)

model.save(model_path)
print(f"Model saved to {model_path}")

test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    test_generator,
    steps=len(test_generator)
)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test AUC: {test_auc}")


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

plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

Image('model_architecture.png')

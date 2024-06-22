from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
import matplotlib.pyplot as plt
import os

model_path = 'vgg16_pneumonia_vs_normal.h5'

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
        return lr * 0.5
    else:
        return lr * 0.1

baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

for layer in baseModel.layers:
    layer.trainable = False

headModel = baseModel.output
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(1, activation="sigmoid")(headModel)

model = models.Model(inputs=baseModel.input, outputs=headModel)

initial_learning_rate = 1e-4
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=initial_learning_rate),
    metrics=["accuracy", Precision(), Recall(), AUC(name='auc')]
)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 32
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
lr_schedule = LearningRateScheduler(adjusted_scheduler)
metrics_callback = MetricsCallback()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=15,
    callbacks=[early_stopping, lr_schedule, metrics_callback]
)

model.save(model_path)
print(f"Model saved to {model_path}")

test_generator = val_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

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
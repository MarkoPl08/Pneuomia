from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import Callback
import os

model_path = 'my_model.keras'

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
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)  # Avoid division by zero

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

model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1, activation='sigmoid')
])


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
    callbacks=[early_stopping, lr_schedule, metrics_callback]  # Add the f1_score_callback to the list
)


model.save('my_model.keras')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_plot.png')

plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss_plot.png')

plt.clf()

plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.savefig('auc_plot.png')
plt.clf()

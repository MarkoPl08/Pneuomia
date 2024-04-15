from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

baseModel = VGG16(weights="imagenet", include_top=False,
    input_tensor=layers.Input(shape=(224, 224, 3)))

for layer in baseModel.layers:
    layer.trainable = False

headModel = baseModel.output
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(256, activation="relu")(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(1, activation="sigmoid")(headModel)

model = models.Model(inputs=baseModel.input, outputs=headModel)

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"])


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
train_generator = train_datagen.flow_from_directory('data/train',
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
val_generator = val_datagen.flow_from_directory('data/val',
                                                target_size=(224, 224),
                                                batch_size=batch_size,
                                                class_mode='binary')

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    epochs=10)

model.save('vgg16_pneumonia_vs_normal.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy with VGG16')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('vgg16_accuracy_plot.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss with VGG16')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('vgg16_loss_plot.png')
plt.clf()

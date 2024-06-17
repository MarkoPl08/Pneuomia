import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_path = 'my_model_functional.keras'


def preprocess_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def visualize_feature_maps(model, img_path):
    img = preprocess_image(img_path)
    model.predict(np.zeros((1, 150, 150, 3)))

    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]

    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)

    images_per_row = 8
    for layer_name, layer_activation in zip(
            [layer.name for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name], activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std() + 1e-5)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()


loaded_model = load_model(model_path)
visualize_feature_maps(loaded_model, 'data/test/PNEUMONIA/person1_virus_6.jpeg')

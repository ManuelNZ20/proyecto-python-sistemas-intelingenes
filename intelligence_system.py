import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Para asegurar reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Redimensionar las imágenes para añadir una dimensión de canal
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalizar las imágenes a valores en el rango [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
# Primera capa convolucional
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# Segunda capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Tercera capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Aplanar los resultados para pasar a las capas densas
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],)

history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Precisión en el conjunto de prueba: {test_acc}")

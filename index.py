import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tqdm import tqdm  # Importar tqdm para la barra de progreso

# Directorio donde se encuentran las imágenes de tu conjunto de datos
data_dir = 'data/caltech-101/101_ObjectCategories'

# Cargar el modelo preentrenado
model = tf.keras.models.load_model('model/transfer_learning_model.keras')

# Crear un extractor de características a partir del modelo
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Listas para almacenar las características y las rutas de las imágenes
features_list = []
image_paths = []

# Contar cuántas imágenes hay para configurar la barra de progreso
total_images = sum([len(files) for r, d, files in os.walk(data_dir)])

# Extraer características de todas las imágenes en las subcarpetas del directorio especificado
for root, dirs, files in os.walk(data_dir):
    for file in tqdm(files, total=total_images, desc="Procesando imágenes"):  # Añadir barra de progreso aquí
        if file.endswith(('png', 'jpg', 'jpeg', 'gif')):
            img_path = os.path.join(root, file)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalización si es necesario
            
            features = feature_extractor.predict(img_array)
            features_list.append(features.flatten())
            
            # Guardar rutas relativas desde 'data/caltech-101/101_ObjectCategories'
            relative_img_path = os.path.relpath(img_path, data_dir)
            image_paths.append(relative_img_path)

# Convertir la lista de características en un array de numpy
features_array = np.array(features_list)

# Guardar las características y las rutas de las imágenes en archivos .npy
np.save('feature_index.npy', features_array)
np.save('image_paths.npy', image_paths)

print("Índice de características guardado en 'feature_index.npy' y rutas de imágenes en 'image_paths.npy'.")

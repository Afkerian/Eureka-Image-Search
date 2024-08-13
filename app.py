from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

# Configuraciones iniciales
app = Flask(__name__)
app.config['IMAGE_FOLDER'] = 'data/caltech-101/101_ObjectCategories'  # Directorio base de las imágenes
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Cargar el modelo
model = tf.keras.models.load_model('model/transfer_learning_model.keras')

# Cargar las clases desde el archivo JSON
with open('class_labels.json', 'r') as json_file:
    class_labels = json.load(json_file)

# Cargar el índice de características
feature_index = np.load('feature_index.npy', allow_pickle=True)
image_paths = np.load('image_paths.npy', allow_pickle=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Ajusta según el tamaño esperado por tu modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalización si es necesario
    return img_array

def extract_features(img_array):
    # Extrae características de la imagen utilizando el modelo (sin la capa final de predicción)
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = feature_extractor.predict(img_array)
    return features

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Verificar si el post request tiene el archivo
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_array = prepare_image(filepath)
            
            # Realizar predicción
            features = extract_features(img_array)
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = class_labels[predicted_class_index]
            
            # Buscar imágenes más similares
            similarities = cosine_similarity(features, feature_index)
            top_10_indices = np.argsort(similarities[0])[::-1][:10]
            
            # Normalizar las rutas de las imágenes para que usen barras normales (/)
            top_10_images = [image_paths[i].replace('\\', '/') for i in top_10_indices]
            
            # Devuelve la predicción y las imágenes similares al usuario
            return render_template('result.html', prediction=predicted_class_name, similar_images=top_10_images)
    return render_template('index.html')

@app.route('/data/caltech-101/101_ObjectCategories/<path:filename>')
def send_file(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

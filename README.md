# Eureka - Image Similarity Search

Eureka es una aplicación web que permite a los usuarios cargar imágenes y obtener las predicciones del modelo de la clase a la que pertenece la imagen, además de mostrar imágenes similares a la imagen cargada. La aplicación está construida con Flask y TensorFlow, y presenta un diseño moderno y atractivo utilizando Bootstrap.

## Características

- **Carga de Imágenes**: Los usuarios pueden cargar una imagen arrastrándola y soltándola o seleccionándola desde su dispositivo.
- **Predicción de Clase**: La aplicación utiliza un modelo preentrenado de TensorFlow para predecir la clase de la imagen cargada.
- **Imágenes Similares**: Se muestran las 10 imágenes más similares a la imagen cargada.
- **Diseño Moderno**: La interfaz de usuario está construida utilizando Bootstrap, con un tema oscuro y un fondo degradado.

## Requisitos

Para ejecutar esta aplicación, necesitarás tener instalados los siguientes paquetes de Python:

```plaintext
Flask 
tensorflow
Pillow
scikit-learn
tqdm
```

Puedes instalar todas las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
eureka/
│
├── data/
│   └── caltech-101/ (contiene las imágenes del conjunto de datos)
├── model/
│   └── transfer_learning_model.keras (modelo de TensorFlow)
├── templates/
│   ├── index.html (página de carga de imágenes)
│   └── result.html (página de resultados de predicción)
├── static/
│   └── css/
│       └── style.css (estilos personalizados si es necesario)
├── uploads/ (directorio donde se guardan las imágenes subidas)
├── app.py (archivo principal de la aplicación Flask)
├── index.py (script para generar el índice de características de las imágenes)
├── requirements.txt (lista de dependencias de Python)
└── README.md (este archivo)
```

## Instrucciones de Uso

### 1. Preparar el Entorno

Clona este repositorio y navega al directorio del proyecto:

```bash
git clone https://github.com/tu-usuario/eureka.git
cd eureka
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

### 2. Generar el Índice de Características

Antes de ejecutar la aplicación, es necesario generar el índice de características de las imágenes en el conjunto de datos. Ejecuta el script `index.py` para generar `feature_index.npy` y `image_paths.npy`:

```bash
python index.py
```

### 3. Ejecutar la Aplicación

Para iniciar la aplicación Flask, ejecuta:

```bash
python app.py
```

Luego, abre tu navegador y navega a `http://127.0.0.1:5000/` para utilizar la aplicación.

### 4. Uso de la Aplicación

- **Subir una Imagen**: En la página de inicio, puedes arrastrar y soltar una imagen o hacer clic en el área de carga para seleccionar una imagen desde tu dispositivo.
- **Ver Resultados**: Una vez que se cargue la imagen, la aplicación mostrará la clase predicha y las imágenes más similares.

## Archivos Clave

### `app.py`

Este archivo contiene la lógica principal de la aplicación Flask, incluyendo la carga de imágenes, la predicción de clase utilizando TensorFlow, y la generación de las rutas de las imágenes similares.

### `index.py`

Este script es responsable de generar las características de las imágenes y guardarlas en un archivo `.npy` para su uso en la aplicación.

### `index.html` y `result.html`

Estos archivos contienen las plantillas HTML para la página de carga de imágenes y la página de resultados, respectivamente. Ambas páginas están diseñadas con Bootstrap para ofrecer una experiencia de usuario moderna y responsiva.

### `class_labels.json`

Contiene las etiquetas de las clases para las predicciones del modelo.

## Créditos

Esta aplicación fue desarrollada por [tu nombre o equipo] como parte de un proyecto de [tu curso/empresa]. Se utilizaron tecnologías como Flask, TensorFlow, y Bootstrap para crear una experiencia de usuario atractiva y funcional.

## Licencia

Este proyecto está licenciado bajo la [licencia de tu elección]. Consulta el archivo `LICENSE` para más detalles.

---

Este README proporciona una descripción completa de tu proyecto, incluidos los pasos necesarios para configurar y ejecutar la aplicación, la estructura del proyecto, y los archivos clave. Si necesitas más ajustes o deseas agregar más detalles, estaré encantado de ayudarte.
import requests
import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
import json
import pandas as pd



load_dotenv()
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')  # URL del backend de FastAPI
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}  # Permitir tanto CSV como Excel

app = Flask(__name__)

# Verifica si el archivo tiene una extensión permitida
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('predecir.html', results=[])

all_predictions = []

@app.route('/predict', methods=['POST'])
def predict():
    global all_predictions
    # Obtener las opiniones desde el formulario
    opiniones = request.form['opinion'].splitlines()
    
    # Filtrar opiniones vacías
    opiniones = [opinion.strip() for opinion in opiniones if opinion.strip()]
    
    # Enviar las opiniones al backend de predicciones
    payload = [{'Textos_espanol': opinion} for opinion in opiniones]
    
    # Supón que tu backend está en el siguiente endpoint
    response = requests.post('http://localhost:8000/predict', json=payload)
    predictions = response.json()  # Obtener las predicciones desde el backend
    
    # Preparar resultados
    classes = ['ODS 3', 'ODS 4', 'ODS 5']
    for i, prediction in enumerate(predictions):
        texto_original = prediction['Textos_espanol']
        pred_num = [round(num * 100, 0) for num in prediction['prediccion']]  # Convertir a porcentaje
        predicted_class = pred_num.index(max(pred_num)) + 1
        
        # Añadir los resultados a la lista global
        all_predictions.append({
            'texto': texto_original,
            'pred_num': pred_num,
            'predicted_class': predicted_class
        })
    
    # Renderizar la página con todas las predicciones hasta ahora
    return render_template('predecir.html', results=all_predictions, num_classes=len(classes), classes=classes)


# Endpoint para la página de entrenamiento
@app.route('/train')
def train():
    return render_template('entrenar.html')

# Endpoint para reentrenar el modelo
@app.route('/retrain', methods=['POST'])
def retrain():
    endpoint = f'{BACKEND_URL}/retrain'

    # Verificar si se ha enviado un archivo
    if 'train_data' not in request.files:
        return render_template('entrenar.html', error='No se ha seleccionado un archivo')

    file = request.files['train_data']

    # Verificar si el archivo no está vacío
    if file.filename == '':
        return render_template('entrenar.html', error='No se encontró el archivo.')

    # Verificar que el archivo sea CSV o Excel
    if not allowed_file(file.filename):
        return render_template('entrenar.html', error='El archivo no es un CSV o Excel.')

    # Verificar la extensión del archivo
    if file.filename.rsplit('.', 1)[1].lower() == 'xlsx':
        # Procesar archivo Excel
        file_df = pd.read_excel(file, engine='openpyxl')
    elif file.filename.rsplit('.', 1)[1].lower() == 'csv':
        # Procesar archivo CSV
        file_df = pd.read_csv(file, encoding='latin1')
    else:
        return render_template('entrenar.html', error='Formato de archivo no soportado.')

    # Verificar que las columnas existan
    if 'sdg' not in file_df.columns or 'Textos_espanol' not in file_df.columns:
        return render_template('entrenar.html', error='El archivo no tiene las columnas requeridas.')

    # Convertir los datos
    payload = file_df.to_dict(orient='records')
    print(payload)

    # Enviar los datos al backend para reentrenamiento
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return render_template('entrenar.html', error=f'Error al comunicarse con el backend: {e}')

    # Obtener la respuesta del backend y procesar los resultados
    try:
        # Obtener los resultados del backend
        model_results = response.json()
        metricas = model_results['metricas']
        palabras_frecuentes = model_results['palabras_frecuentes'] 

        # Obtener las métricas
        accuracy = metricas['accuracy']
        recall = metricas['recall']
        precision = metricas['precision']
        f1_score = metricas['f1_score']

        # Formatear las métricas
        formatear = {
            'accuracy': f"{accuracy:.2%}",
            'recall': f"{recall:.2%}",
            'precision': f"{precision:.2%}",
            'f1_score': f"{f1_score:.2%}"
        }

        # Renderizar las métricas y las palabras frecuentes en la página
        return render_template('entrenar.html', 
                               metricas=formatear, 
                               palabras_frecuentes=palabras_frecuentes) 
    
    except Exception as e:
        return render_template('entrenar.html', error=f"Error procesando la respuesta del backend: {e}")

if __name__ == '__main__':
    app.run(debug=True)
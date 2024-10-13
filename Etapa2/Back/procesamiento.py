import os
import pandas as pd
import ftfy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


import re
import joblib

import nltk
from nltk.tokenize import word_tokenize

import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
#----------------------------------------------------------------
# Carga datos
def cargar_datos_excel(excel_path):
    datos = pd.read_excel(excel_path, engine='openpyxl')
    return datos

def cargar_datos_csv(csv_path):
    datos = pd.read_csv(csv_path, encoding='latin1') 
    return datos

# Cambiar a CVS
def guardar_datos_csv(datos, output_path):
    datos.to_csv(output_path, index=False, encoding='utf-8')

# Corregir codificación texto
def corregir_codificacion(datos, columna):
    datos[columna] = datos[columna].apply(ftfy.fix_text)
    return datos

def tokenizer(text):
    # Convertir texto a minúsculas y corregir caracteres extraños
    text = text.lower()
    text = ftfy.fix_text(text)
    
    # Tokenizar
    words = word_tokenize(text)
    
    # Quitar puntuación
    words = [re.sub(r'[^\w\s]', '', word) for word in words if word]

    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]

    # Lematizar verbos
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words]

    return words

# imprimir palabras más frecuentes por categoría
def print_most_frequent_words(pipeline, datos, columna_ods, categorias_ods, n=10):
    tfidf_vectorizer = pipeline.named_steps['tfidf']
    palabras_frecuentes_por_ods = {}
    
    for ods in categorias_ods:
        texts_for_ods = datos[datos[columna_ods] == ods]['Textos_espanol']
        X = tfidf_vectorizer.transform(texts_for_ods)
        words = tfidf_vectorizer.get_feature_names_out()
        sums = X.sum(axis=0).A1
        word_freq = [(word, sums[idx]) for idx, word in enumerate(words)]
        word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
        palabras_frecuentes_por_ods[ods] = word_freq[:n] 
    
    return palabras_frecuentes_por_ods
        
#----------------------------------------------------------------


#Evaluación de metricas del modelo de clasificación
def evaluar_modelo(y_test, y_pred):

    print('Exactitud: %.2f' % accuracy_score(y_test, y_pred))
    print("Recall: {}".format(recall_score(y_test, y_pred, average='macro')))
    print("Precisión: {}".format(precision_score(y_test, y_pred, average='macro')))
    print("Puntuación F1: {}".format(f1_score(y_test, y_pred, average='macro')))

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Creación Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)


# Pipeline de procesamiento de datos
def pipeline_procesamiento_datos(data, columna_texto, columna_ods):
   
    if isinstance(data, str) and data.endswith('.csv'):
        datos = cargar_datos_csv(data)
    elif isinstance(data, str) and data.endswith('.xlsx'):
        datos = cargar_datos_excel(data)
    elif isinstance(data, pd.DataFrame):
        datos = data
    else:
        raise ValueError("El tipo de 'data' no es válido. Debe ser una ruta de archivo o un DataFrame.")
    
    # Creación del pipeline de preprocesamiento y modelo
    pipeline = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(tokenizer=tokenizer)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=0))
    ])

    # Dividir los datos en características (X) y etiquetas (y)
    X_data = datos[columna_texto]
    y_data = datos[columna_ods]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Ajustar el pipeline
    pipeline.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = pipeline.predict(X_test)

    # Evaluar el modelo
    evaluar_modelo(y_test, y_pred)

    # Imprimir las palabras más frecuentes para cada categoría
    frecuentes = print_most_frequent_words(pipeline, datos, columna_ods='sdg', categorias_ods=[3, 4, 5], n=10)


    # calcular metricas 
    metricas = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro')
    }

    joblib.dump(pipeline, 'pipeline.joblib')

    return {
        'metricas': metricas, 
        'frecuentes': frecuentes,
    }
# import necessary libraries
import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
import procesamiento as procesamiento
from joblib import load
from procesamiento import tokenizer, pipeline_procesamiento_datos
from pydantic import BaseModel
from typing import List

# MODELOS DE DATOS
class DataModel(BaseModel):
    # opiniones
    Textos_espanol: str
    def get_column_names(self):
        return ["Textos_espanol"]
    
class TrainModel(BaseModel):
    # opiniones
    Textos_espanol: str  
    # la categoría ODS como variable objetivo
    sdg: int  

    def get_column_names(self):
        return ["Textos_espanol", "sdg"]

class PredictionResult(BaseModel):
    Textos_espanol: str
    prediccion: List[float] 

    #columnas de la predicción.
    def get_column_names(self):
        return ["Textos_espanol", "prediccion"]

# CLASE DEL MODELO
class Model:
    def __init__(self):
        # pipline de procesamiento de datos preentrenado
        self.model = load("pipeline.joblib")


    # Método para hacer predicciones
    def predicciones(self, data: DataFrame):
        print(f"Prediciendo {data.shape[0]} muestras")
        predictions = []
        # Itera sobre las filas del DataFrame
        for index, row in data.iterrows():
            texto = row["Textos_espanol"]

            try:
                # Realiza la predicción de las probabilidades de cada categoría
                probabilidades = self.model.predict_proba([texto])
                prediccion = probabilidades[0].tolist()
                resultado = {"Textos_espanol": texto, "prediccion": prediccion} 
                predictions.append(resultado)
            # Manejo de errores
            except Exception as e:
                print(f"Error de prediccion: {e}")
                raise e

        return predictions



    # Método para reentrenar el modelo con nuevos datos
    def retrain(self, data_inicial, columna_texto, columna_ods):
        print(f"Reentrenando el modelo con datos proporcionados.")
        try:
            # Cargar y procesar los datos de entrenamiento utilizando el pipeline
            resultado = pipeline_procesamiento_datos(data_inicial, columna_texto, columna_ods)
            
            # Imprimir el resultado completo para verificar
            print(f"Resultado del pipeline: {resultado}")
            
            # Guardar el pipeline reentrenado
            print(f"Guardando el modelo reentrenado.")
            joblib.dump(self.model, 'modelo.joblib')

        except Exception as e:
            print(f"Error durante el reentrenamiento: {e}")
            raise e

        return resultado
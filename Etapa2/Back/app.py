from model import Model, DataModel, TrainModel
from fastapi import FastAPI
from pandas import DataFrame


app = FastAPI()

#listas utilizadas en los endpoints
DataModelArray = list[DataModel]
TrainModelArray = list[TrainModel]

# Endpoint rectificar que la API está corriendo
@app.get("/")
def read_root():
   return {"message": "API is running"}


# Endpoint para predecir, donde se recibe un array de instancias de DataModel
@app.post("/predict")
def predicciones(data: DataModelArray):
    # Convertir el array de objetos en un DataFrame
    dataInicial = DataFrame([item.dict() for item in data])
    # Crear una instancia del modelo
    modelo = Model()
    prediccion = modelo.predicciones(dataInicial)
    return prediccion

# Endpoint para reentrenar el modelo, donde se recibe un array de instancias de TrainModel
@app.post("/retrain")
def retrain_model(data: TrainModelArray):
    try:
        # Convertir los datos recibidos en un DataFrame
        data_inicial = DataFrame([item.dict() for item in data])
        print(f"Datos recibidos para reentrenar: \n{data_inicial.head()}")

        # Crear una instancia del modelo
        model = Model()

        # Reentrenar el modelo
        resultado = model.retrain(data_inicial, "Textos_espanol", "sdg")
        
        # métricas del reentrenamiento
        return {
            "metricas": resultado["metricas"],
            "palabras_frecuentes": resultado["frecuentes"]
        }
    
    except KeyError as ke:
        print(f"Error en las métricas durante el reentrenamiento: {ke}")
        raise ValueError("Error en los datos de las métricas") from ke
    except Exception as e:
        print(f"Error en el endpoint /retrain: {e}")
        raise RuntimeError("Error durante el reentrenamiento") from e



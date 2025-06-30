import pandas as pd
import numpy as np
import joblib
import os

print("--- Inicio del script de predicción ---")

# Definir Rutas 
DATA_PATH = r"C:\Users\Usuario\Documents\TheBridge\REPO_DE_TRABAJO\proyecto ML\proyecto\src\data\raw\test.csv"
MODEL_PATH = r"C:\Users\Usuario\Documents\TheBridge\REPO_DE_TRABAJO\proyecto ML\proyecto\src\model\my_model.joblib"
SUBMISSION_PATH = r"C:\Users\Usuario\Documents\TheBridge\REPO_DE_TRABAJO\proyecto ML\proyecto\data\processed\submission.csv"

# Cargar el Modelo y los Datos Nuevos
print(f"Cargando modelo desde: {MODEL_PATH}")
try:
    model_payload = joblib.load(MODEL_PATH)
    model = model_payload['model']
    model_columns = model_payload['columns']
except FileNotFoundError:
    print(f"Error: No se encontró el modelo. Ejecuta train.py primero.")
    exit()

print(f"Cargando datos nuevos desde: {DATA_PATH}")
try:
    test_df = pd.read_csv(DATA_PATH)
    test_ids = test_df['Id']
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de test.")
    exit()

#  Preprocesar los Datos Nuevos 
print("Preprocesando datos nuevos...")
X_test = test_df.drop('Id', axis=1)

# Imputación
columnas_numericas = X_test.select_dtypes(include=np.number).columns
columnas_categoricas = X_test.select_dtypes(include='object').columns
for col in columnas_numericas:
    X_test[col].fillna(X_test[col].median(), inplace=True)
for col in columnas_categoricas:
    X_test[col].fillna(X_test[col].mode()[0], inplace=True)

# One-Hot Encoding y alineación de columnas
X_test_processed = pd.get_dummies(X_test)
final_test_df = pd.DataFrame(columns=model_columns)
final_test_df = pd.concat([final_test_df, X_test_processed], ignore_index=True, sort=False).fillna(0)
final_test_df = final_test_df[model_columns]

#  Realizar Predicciones 
print("Realizando predicciones...")
predictions_log = model.predict(final_test_df)
final_predictions = np.expm1(predictions_log)

#  Generar Archivo de Submission 
print(f"Guardando predicciones en: {SUBMISSION_PATH}")
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_predictions})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ ¡Archivo de submission creado exitosamente!")
print("--- Fin del script de predicción ---")
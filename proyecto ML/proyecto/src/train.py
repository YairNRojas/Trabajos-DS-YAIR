import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

print("--- Inicio del script de entrenamiento ---")

#  Definir Rutas (¡CORREGIDO para la nueva carpeta 'proyecto'!) ---
DATA_PATH = r"C:\Users\Usuario\Documents\TheBridge\REPO_DE_TRABAJO\proyecto ML\proyecto\src\data\raw\train.csv"
MODEL_DIR = r"C:\Users\Usuario\Documents\TheBridge\REPO_DE_TRABAJO\proyecto ML\proyecto\src\model"
MODEL_PATH = os.path.join(MODEL_DIR, 'my_model.joblib')

#  Cargar los Datos
print(f"Cargando datos desde: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH)
    print("Datos cargados correctamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en {DATA_PATH}. Verifica la estructura de carpetas.")
    exit()

# Preparación de los Datos
print("Iniciando preprocesamiento de los datos...")
X = df.drop(['SalePrice', 'Id'], axis=1)
y_log = np.log1p(df['SalePrice'])

# Imputación de valores nulos
columnas_numericas = X.select_dtypes(include=np.number).columns
columnas_categoricas = X.select_dtypes(include='object').columns
for col in columnas_numericas:
    X[col].fillna(X[col].median(), inplace=True)
for col in columnas_categoricas:
    X[col].fillna(X[col].mode()[0], inplace=True)
print("Imputación de valores nulos completada.")

# One-Hot Encoding
X_processed = pd.get_dummies(X)
print("Codificación de variables categóricas completada.")

# Entrenamiento del Modelo Final
print("Entrenando el modelo final XGBoost Regressor...")
final_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=42)
final_model.fit(X_processed, y_log, verbose=False)
print("¡Modelo entrenado exitosamente!")

#  Guardar el Modelo Entrenado
print(f"Guardando el modelo en: {MODEL_PATH}")
os.makedirs(MODEL_DIR, exist_ok=True)
model_payload = {
    'model': final_model,
    'columns': X_processed.columns
}
joblib.dump(model_payload, MODEL_PATH)
print(f"✅ ¡Modelo guardado correctamente como '{os.path.basename(MODEL_PATH)}'!")
print("--- Fin del script de entrenamiento ---")
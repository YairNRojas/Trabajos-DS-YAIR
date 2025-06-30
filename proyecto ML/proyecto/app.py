import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------------------------------------------------------
# Configuración de la Página y Carga del Modelo
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Predictor de Precios de Viviendas", page_icon="🏠", layout="wide")

# Usar caché para cargar el modelo solo una vez y mejorar el rendimiento
@st.cache_data
def load_model():
    """Carga el modelo y las columnas desde el archivo guardado."""
    # La ruta es relativa a la ubicación de app.py
    model_path = 'src/model/my_model.joblib'
    try:
        payload = joblib.load(model_path)
        model = payload['model']
        model_columns = payload['columns']
        return model, model_columns
    except FileNotFoundError:
        return None, None

# Cargar los artefactos del modelo
model, model_columns = load_model()

# -----------------------------------------------------------------------------
# Interfaz de Usuario (Sidebar para introducir datos)
# -----------------------------------------------------------------------------
st.sidebar.header('Introduzca las Características de la Vivienda')

def user_input_features():
    """Crea los widgets en la barra lateral para recoger la entrada del usuario."""
    overall_qual = st.sidebar.slider('Calidad General (OverallQual)', 1, 10, 7)
    gr_liv_area = st.sidebar.slider('Área Habitable (GrLivArea) en m²', 50, 400, 150) # Asumiendo m² para slider
    garage_cars = st.sidebar.selectbox('Capacidad del Garaje (Coches)', [0, 1, 2, 3, 4], 2)
    total_bsmt_sf = st.sidebar.slider('Área del Sótano (TotalBsmtSF) en m²', 0, 300, 100) # Asumiendo m²
    year_built = st.sidebar.slider('Año de Construcción (YearBuilt)', 1870, 2010, 2005)

    data = {
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area * 10.764, # Convertir m² a pies cuadrados si el modelo lo espera así
        'GarageCars': garage_cars,
        'TotalBsmtSF': total_bsmt_sf * 10.764, # Convertir m² a pies cuadrados
        'YearBuilt': year_built
    }
    return pd.DataFrame(data, index=[0])

user_df = user_input_features()

# -----------------------------------------------------------------------------
# Lógica Principal y Visualización en la Página
# -----------------------------------------------------------------------------
st.title('🏠 Simulador de Precios de Viviendas')
st.write("""
Esta aplicación utiliza un modelo de **XGBoost** para predecir el precio de una vivienda.
Modifica los valores en el panel de la izquierda para ver el precio estimado al instante.
""")

if model is None:
    st.error("⚠️ Modelo no encontrado en `src/model/my_model.joblib`. Por favor, asegúrate de haber ejecutado `src/train.py` primero.")
else:
    # Preparar el DataFrame de entrada para que coincida con el formato del modelo
    input_df = pd.DataFrame(columns=model_columns)
    input_df = pd.concat([input_df, user_df], ignore_index=True, sort=False).fillna(0)
    
    # Asegurar el orden y la existencia de todas las columnas
    input_df = input_df[model_columns]

    # Realizar la predicción
    prediction_log = model.predict(input_df)
    prediction = np.expm1(prediction_log)

    # --- Mostrar la predicción ---
    st.header('Predicción de Precio')
    st.write("---")
    st.metric(label="**Precio Estimado de la Vivienda**", value=f"${prediction[0]:,.2f}")
    st.write("---")

    # Botón de celebración
    if st.button('¡Celebrar Predicción!'):
        st.balloons()
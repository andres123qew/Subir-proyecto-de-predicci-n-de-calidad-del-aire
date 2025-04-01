import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo entrenado y el escalador
scaler_model = joblib.load("scaler_model.bin")
knn_trained_model = joblib.load("knn_trained_model.bin")

# Título y encabezado
st.title("Aplicación de Predicción de la Calidad del Aire")
st.subheader("Desarrollado por [Tu Nombre]")

# Mostrar una imagen representativa
st.image("https://www.poligonosindustrialesasturias.com/udecontrol_datos/objetos/2283.jpg", use_column_width=True)

# Barra lateral para ingresar datos
st.sidebar.header("Introduzca los valores del AQI")
aqi_pm25 = st.sidebar.slider("AQI PM2.5", min_value=-1.0, max_value=1.0, value=0.0)
aqi_co = st.sidebar.slider("AQI CO", min_value=-1.0, max_value=1.0, value=0.0)
aqi_no2 = st.sidebar.slider("AQI NO2", min_value=-1.0, max_value=1.0, value=0.0)

# Crear un DataFrame con los valores proporcionados por el usuario
input_data = pd.DataFrame([[aqi_pm25, aqi_co, aqi_no2]], columns=["PM2.5 AQI", "CO AQI", "NO2 AQI"])

# Normalizar los datos usando el escalador
normalized_data = scaler_model.transform(input_data)

# Hacer la predicción
prediction_result = knn_trained_model.predict(normalized_data)[0]

# Mostrar los resultados de la predicción
if prediction_result == 0:
    st.markdown("<h3 style='color: green;'>La calidad del aire es segura</h3>", unsafe_allow_html=True)
    st.markdown("✔️ El aire no presenta riesgos.")
else:
    st.markdown("<h3 style='color: red;'>Alto riesgo de aire contaminado</h3>", unsafe_allow_html=True)
    st.markdown("⚠️ Precaución: La calidad del aire es peligrosa.")

# Agregar una línea divisoria
st.markdown("---")

# Información de copyright en el pie de página
st.markdown("© Universidad Autónoma de Bucaramanga (UNAB) 2025")

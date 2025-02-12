import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo y el escalador
model = joblib.load("modelo_regresion.bin")
scaler = joblib.load("scaler.bin")

# Configuración de la aplicación
st.title("Predictor de compra de bici")
st.write("### Autor: ChatGPT y juandi")

# Sidebar con los inputs de usuario
st.sidebar.header("Ingrese los valores")

edad = st.sidebar.slider("Edad", min_value=20, max_value=60, value=40, step=1)
ingresos = st.sidebar.slider("Ingresos", min_value=30000, max_value=150000, value=80000, step=1000)
experiencia = st.sidebar.selectbox("Experiencia", options=list(range(1, 14)), index=6)
satisfaccion = st.sidebar.selectbox("Satisfacción", options=list(range(3, 11)), index=3)

# Mostrar los valores ingresados
st.write("### Valores ingresados")
st.write(f"- **Edad:** {edad} años")
st.write(f"- **Ingresos:** ${ingresos}")
st.write(f"- **Experiencia:** {experiencia} años")
st.write(f"- **Satisfacción:** {satisfaccion}/10")

# Normalización de los datos
input_data = np.array([[edad, ingresos, experiencia, satisfaccion]])
input_scaled = scaler.transform(input_data)

# Predicción
prediccion = model.predict(input_scaled)[0]

# Mostrar el resultado
st.write("### Valor de compra predicho")
st.write(f"## ${prediccion:,.2f}")

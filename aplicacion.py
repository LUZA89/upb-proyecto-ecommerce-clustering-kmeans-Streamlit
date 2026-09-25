"""
App de Streamlit: clasifica una transacción de Olist en uno de los 3 clústeres
(entregas de proximidad, envíos regionales o envíos de larga distancia) y
sugiere el nivel de prioridad de seguimiento logístico.

Archivos que debe tener en la misma carpeta:
- escalador.pkl       -> MinMaxScaler ya ajustado con los datos de entrenamiento
- modelo_kmeans.pkl   -> modelo KMeans (k=3) ya entrenado
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# 1) Carga del escalador y el modelo entrenados
# ------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    with open("escalador.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("modelo_kmeans.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = cargar_modelo()

VARIABLES = ["total_payment_value", "total_freight", "total_weight_g",
             "total_volume_cm3", "distancia_km"]

# Perfil de cada clúster (calculado en el notebook: número = model.predict)
# OJO: el número de clúster que asigna KMeans es arbitrario; aquí se mapea
# al nombre correcto según los centroides ya conocidos.
PERFILES = {
    0: {"nombre": "Entregas de Proximidad", "prioridad": "Baja",
        "color": "🟢", "descripcion": "Distancia corta, flete y pago bajos."},
    1: {"nombre": "Envíos de Larga Distancia", "prioridad": "Alta",
        "color": "🔴", "descripcion": "Distancia larga y flete alto. Mayor riesgo de demora."},
    2: {"nombre": "Envíos Regionales", "prioridad": "Media",
        "color": "🟠", "descripcion": "Distancia media; los pedidos más pesados y voluminosos."},
}

# ------------------------------------------------------------------
# 2) Interfaz
# ------------------------------------------------------------------
st.set_page_config(page_title="Clustering Olist", page_icon="📦")
st.title("📦 Segmentación de transacciones de Olist")
st.write(
    "Ingresa los datos de un pedido para identificar a qué grupo pertenece "
    "y qué nivel de prioridad de seguimiento logístico le corresponde."
)

col1, col2 = st.columns(2)
with col1:
    pago = st.number_input("Valor del pago (R$)", min_value=0.0, value=100.0, step=1.0)
    flete = st.number_input("Costo del flete (R$)", min_value=0.0, value=15.0, step=1.0)
    peso = st.number_input("Peso total (g)", min_value=0.0, value=1500.0, step=50.0)
with col2:
    volumen = st.number_input("Volumen total (cm³)", min_value=0.0, value=12000.0, step=100.0)
    distancia = st.number_input("Distancia vendedor-cliente (km)", min_value=0.0, value=200.0, step=10.0)

if st.button("Clasificar pedido"):
    entrada = pd.DataFrame([[pago, flete, peso, volumen, distancia]], columns=VARIABLES)
    entrada_norm = scaler.transform(entrada)
    cluster = int(model.predict(entrada_norm)[0])
    perfil = PERFILES[cluster]

    st.subheader(f"{perfil['color']} Grupo: {perfil['nombre']}")
    st.write(f"**Prioridad de seguimiento logístico:** {perfil['prioridad']}")
    st.write(perfil["descripcion"])

    with st.expander("Ver detalle técnico"):
        st.write("Datos ingresados (normalizados 0 a 1):")
        st.dataframe(pd.DataFrame(entrada_norm, columns=VARIABLES).round(3))
        distancias = model.transform(entrada_norm)[0]
        st.write("Distancia a cada centroide (más bajo = más cercano):")
        st.dataframe(pd.DataFrame([distancias], columns=[f"Clúster {i}" for i in range(len(distancias))]).round(3))

st.divider()
st.caption(
    "Modelo K-Means (k=3) entrenado sobre 25.516 transacciones de Olist Store, "
    "usando pago, flete, peso, volumen y distancia. Ver el notebook del análisis "
    "completo en el repositorio."
)

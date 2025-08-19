
# Hacer una aplicación Streamlit para predecir el precio de Binance Coin (BNB) con el precio de Ethereum de hace 15 días
import streamlit as st
import pandas as pd
import numpy as np

# Cargar el modelo entrenado
from pickle import load
model = load(open('../models/best_model.pkl', 'rb'))
scaler = load(open('../models/scaler.pkl', 'rb'))
# Cargar el scaler de Binance Coin
scaler_binance = load(open('../models/scaler_binance.pkl', 'rb'))

# hacer un sidebar para seleccionar el valor de Ethereum
st.sidebar.header('Parámetros de entrada')
eth_price = st.sidebar.slider('Precio de Ethereum (USD) hace 15 días', min_value=0.0, max_value=5000.0, value=2000.0, step=10.0)
eth_price = np.array(eth_price).reshape(-1, 1)
# escalar el valor de Ethereum
eth_price_scaled = scaler.transform(eth_price)

# predecir el precio de Binance Coin
bnb_price_scaled = model.predict(eth_price_scaled)
# desescalar el valor de Binance Coin
bnb_price = scaler_binance.inverse_transform(bnb_price_scaled.reshape(-1, 1))
st.write(f'El precio predicho de Binance Coin (BNB) es: ${bnb_price[0][0]}')
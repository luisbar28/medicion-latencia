import time
import socket
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from scapy.all import sr1, IP, ICMP
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LinearRegression

# Lista de servidores en la nube para probar latencia
SERVIDORES = {
    "Google Cloud": "8.8.8.8",  # DNS de Google
    "AWS": "52.94.231.248",  # Nueva IP de AWS
    "Azure": "20.50.2.1",  # IP de Azure
}

def medir_latencia(ip):
    """Envía un paquete ICMP (ping) y mide el tiempo de respuesta."""
    try:
        inicio = time.time()
        respuesta = sr1(IP(dst=ip)/ICMP(), timeout=1, verbose=0)
        if respuesta:
            return (time.time() - inicio) * 1000  # Convertir a ms
        else:
            return None  # Sin respuesta
    except Exception as e:
        print(f"Error al medir latencia con {ip}: {e}")
        return None

def analizar_latencias():
    """Mide la latencia de cada servidor en paralelo y muestra resultados."""
    resultados = {}
    
    with ThreadPoolExecutor(max_workers=len(SERVIDORES)) as executor:
        futuros = {executor.submit(lambda s: (s[0], [medir_latencia(s[1]) for _ in range(5)]), item): item for item in SERVIDORES.items()}
        
        for futuro in futuros:
            nombre, latencias = futuro.result()
            latencias = [l for l in latencias if l is not None]  # Filtrar valores None
            if latencias:
                resultados[nombre] = {
                    "Promedio (ms)": float(np.mean(latencias)),
                    "Mínimo (ms)": float(np.min(latencias)),
                    "Máximo (ms)": float(np.max(latencias)),
                }
            else:
                resultados[nombre] = "No responde"
    return resultados

def predecir_latencia(historial, servidor):
    """Usa regresión lineal para predecir la latencia futura."""
    if len(historial) < 3:
        return "Insuficientes datos para predecir"
    
    datos_servidor = [df[df["Servidor"] == servidor]["Promedio (ms)"].values for df in historial if servidor in df["Servidor"].values]
    datos_servidor = [x[0] for x in datos_servidor if len(x) > 0]
    
    if len(datos_servidor) < 3:
        return "Insuficientes datos para predecir"
    
    X = np.arange(len(datos_servidor)).reshape(-1, 1)
    y = np.array(datos_servidor)
    modelo = LinearRegression().fit(X, y)
    prediccion = modelo.predict([[len(datos_servidor)]])
    return round(prediccion[0], 2)

# Historial de mediciones
historial = []

if __name__ == "__main__":
    st.set_page_config(page_title="Medición de Latencia", page_icon="🌐", layout="wide")
    st.title("🌐 Medición de Latencia en la Nube")
    st.write("Este dashboard muestra la latencia en ms de diferentes proveedores de nube.")
    
    st.sidebar.header("Opciones")
    auto_run = st.sidebar.checkbox("Ejecutar medición automáticamente", value=True)
    
    if auto_run or st.button("Iniciar Medición 🔄"):
        st.write("📡 Midiendo latencias...")
        resultados = analizar_latencias()
        
        # Convertir resultados a DataFrame
        data = []
        for servidor, datos in resultados.items():
            if isinstance(datos, dict):
                data.append([servidor, datos["Promedio (ms)"], datos["Mínimo (ms)"], datos["Máximo (ms)"]])
            else:
                data.append([servidor, None, None, None])
        df = pd.DataFrame(data, columns=["Servidor", "Promedio (ms)", "Mínimo (ms)", "Máximo (ms)"])
        
        # Agregar al historial
        historial.append(df)
        if len(historial) > 5:  # Limitar historial a 5 mediciones
            historial.pop(0)
        
        # Mostrar ranking
        st.subheader("🏆 Ranking de Servidores")
        df_sorted = df.sort_values(by="Promedio (ms)", ascending=True).reset_index(drop=True)
        st.dataframe(df_sorted.style.highlight_min(subset=["Promedio (ms)"], color="lightgreen"))
        
        # Mostrar tabla de mediciones
        st.subheader("📊 Resultados de la última medición")
        st.dataframe(df)
        
        # Mostrar gráfico
        fig = px.bar(df, x="Servidor", y=["Promedio (ms)", "Mínimo (ms)", "Máximo (ms)"], barmode="group", title="Comparación de Latencias")
        st.plotly_chart(fig)
        
        # Mostrar historial
        if historial:
            st.subheader("📜 Historial de Mediciones")
            for i, df_hist in enumerate(historial[::-1]):
                st.write(f"Medición {len(historial) - i}:")
                st.dataframe(df_hist)
        
        # Mostrar predicciones
        st.subheader("🔮 Predicción de Latencia")
        predicciones = {servidor: predecir_latencia(historial, servidor) for servidor in SERVIDORES}
        pred_df = pd.DataFrame(list(predicciones.items()), columns=["Servidor", "Predicción de Latencia (ms)"])
        st.dataframe(pred_df)

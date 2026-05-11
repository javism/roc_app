import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc

st.set_page_config(page_title="Simulador ROC Completo", layout="wide")

st.title("App Interactiva: ROC y Matriz de Confusión")

# --- Sidebar: Parámetros ---
st.sidebar.header("Configuración")
media_sanos = st.sidebar.slider("Media Sanos (Azul)", 0.0, 5.0, 2.0, 0.1)
media_enfermos = st.sidebar.slider("Media Enfermos (Rojo)", 0.0, 10.0, 5.0, 0.1)
std = st.sidebar.slider("Desviación Estándar", 0.5, 3.0, 1.2, 0.1)
umbral = st.sidebar.slider("Umbral de Decisión", 0.0, 10.0, 3.5, 0.1)
n_total = st.sidebar.number_input("Tamaño de muestra (para Matriz)", 100, 10000, 1000)

# --- Cálculos de Probabilidad ---
# Sensibilidad (TPR) y Especificidad (TNR)
tpr = 1 - norm.cdf(umbral, media_enfermos, std)
tnr = norm.cdf(umbral, media_sanos, std)
fpr = 1 - tnr
fnr = 1 - tpr

# Cálculo del AUC Teórico para dos normales
# AUC = Phi((mu1 - mu2) / sqrt(sigma1^2 + sigma2^2))
auc_teorico = norm.cdf(abs(media_enfermos - media_sanos) / np.sqrt(2 * std**2))

# --- Layout de Pantalla ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuciones")
    x = np.linspace(-2, 12, 500)
    fig1, ax1 = plt.subplots()
    ax1.plot(x, norm.pdf(x, media_sanos, std), color="blue", label="Sanos")
    ax1.plot(x, norm.pdf(x, media_enfermos, std), color="red", label="Enfermos")
    ax1.axvline(umbral, color="black", linestyle="--", lw=2)
    ax1.fill_between(x, norm.pdf(x, media_sanos, std), where=(x >= umbral), color='blue', alpha=0.3, label="FP")
    ax1.fill_between(x, norm.pdf(x, media_enfermos, std), where=(x < umbral), color='red', alpha=0.3, label="FN")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader(f"Curva ROC (AUC: {auc_teorico:.3f})")
    u_vals = np.linspace(-2, 12, 200)
    tpr_c = [1 - norm.cdf(u, media_enfermos, std) for u in u_vals]
    fpr_c = [1 - norm.cdf(u, media_sanos, std) for u in u_vals]
    
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr_c, tpr_c, color="darkorange", lw=3)
    ax2.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax2.scatter(fpr, tpr, color="black", s=100, zorder=5)
    ax2.set_xlabel("FPR (1 - Especificidad)")
    ax2.set_ylabel("TPR (Sensibilidad)")
    st.pyplot(fig2)

# --- Matriz de Confusión ---
st.divider()
st.subheader("Matriz de Confusión (Valores Estimados)")

# Asumimos prevalencia del 50% para el ejemplo
verdaderos_positivos = int((n_total / 2) * tpr)
falsos_negativos = int((n_total / 2) * fnr)
verdaderos_negativos = int((n_total / 2) * tnr)
falsos_positivos = int((n_total / 2) * fpr)

data = {
    "Predicho Sano (-)": [verdaderos_negativos, falsos_negativos],
    "Predicho Enfermo (+)": [falsos_positivos, verdaderos_positivos]
}
df_matriz = pd.DataFrame(data, index=["Real Sano", "Real Enfermo"])

c1, c2 = st.columns([1, 2])
with c1:
    st.table(df_matriz)
    st.info(f"""
       **Métricas derivadas de la matriz de confusión:** 
       * **TPR/Sensibilidad ( P(Test+ | Enfermo) ): {tpr:.2}**
       * **FNR/Especificidad ( P(Test- | Sano) ): {tnr:.2}**
       * **FPR: {fpr:.2}**
    """)

with c2:
    st.info(f"""
    **Interpretación del test con Umbral {umbral}:**
    * Se detecta correctamente al **{tpr:.1%}** de los enfermos.
    * Se equivoca con el **{fpr:.1%}** de los sanos (falsas alarmas).
    * El **AUC de {auc_teorico:.2f}** indica que hay un {auc_teorico:.1%} de probabilidad de que el test clasifique correctamente a un individuo enfermo elegido al azar frente a uno sano.
    """)

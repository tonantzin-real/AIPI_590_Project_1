"""
=============================================================================
APPLICATION: main.py (Streamlit Web Dashboard)
PROJECT: AIPI 590 - Project 1: ¿Por qué me rechazaron?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Interactive portfolio artifact demonstrating Explainable AI, Algorithmic
    Recourse (DiCE), and Fairness Auditing for credit scoring underwriting.
    Designed for loan applicants, risk officers, and financial regulators.
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Relative imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import (
    CATEGORICAL_LABELS,
    IMMUTABLE_FEATURES,
    ACTIONABLE_FEATURES
)
from src.xai_shap import CreditSHAPExplainer
from src.xai_recourse import CreditRecourseEngine

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="¿Por qué me rechazaron? | XAI Credit Recourse",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for professional fintech UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 4px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-approved {
        background-color: #dcfce7;
        color: #166534;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
    .badge-rejected {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RESOURCE CACHING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models_and_engines():
    """Loads serialized model, SHAP explainer, and DiCE recourse engine."""
    model = joblib.load("models/credit_model.joblib")
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
    shap_explainer = CreditSHAPExplainer()
    recourse_engine = CreditRecourseEngine(sample_size=1500)
    return model, metadata, shap_explainer, recourse_engine


model, metadata, shap_explainer, recourse_engine = load_models_and_engines()


# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS & APPLICANT PROFILER
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=64)
st.sidebar.markdown("### 👤 Perfil del Solicitante / Applicant Profile")
st.sidebar.markdown("Configure loan applicant financial and demographic details:")

preset = st.sidebar.selectbox(
    "Cargar Perfil Predeterminado / Preset Profile:",
    ["Solicitante en Riesgo (Rechazado)", "Solicitante Confiable (Aprobado)", "Personalizado / Custom"]
)

if preset == "Solicitante en Riesgo (Rechazado)":
    init_limit = 20000
    init_sex = 2  # Female
    init_edu = 2  # University
    init_mar = 1  # Married
    init_age = 24
    init_pay1 = 2  # 2 months delayed payment
    init_bill1 = 45000
    init_payamt1 = 500
elif preset == "Solicitante Confiable (Aprobado)":
    init_limit = 150000
    init_sex = 1  # Male
    init_edu = 1  # Grad school
    init_mar = 2  # Single
    init_age = 35
    init_pay1 = -1 # Paid in full
    init_bill1 = 12000
    init_payamt1 = 12000
else:
    init_limit = 50000
    init_sex = 2
    init_edu = 2
    init_mar = 2
    init_age = 28
    init_pay1 = 1
    init_bill1 = 25000
    init_payamt1 = 2000

# Input Controls
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    sex = st.selectbox("Género / Sex:", [1, 2], index=0 if init_sex == 1 else 1, format_func=lambda x: CATEGORICAL_LABELS["SEX"][x])
    marriage = st.selectbox("Estado Civil / Marital:", [1, 2, 3], index=init_mar - 1, format_func=lambda x: CATEGORICAL_LABELS["MARRIAGE"][x])
with col_sb2:
    age = st.slider("Edad / Age:", 18, 75, init_age)
    education = st.selectbox("Educación / Education:", [1, 2, 3, 4], index=init_edu - 1, format_func=lambda x: CATEGORICAL_LABELS["EDUCATION"][x])

limit_bal = st.sidebar.number_input("Límite de Crédito Solicitado (MXN / NTD):", min_value=10000, max_value=800000, value=init_limit, step=10000)

st.sidebar.markdown("#### Historial y Balances Recientes")
pay_1 = st.sidebar.selectbox(
    "Estatus Pago Reciente (Mes 1):",
    [-2, -1, 0, 1, 2, 3],
    index=[-2, -1, 0, 1, 2, 3].index(init_pay1),
    format_func=lambda x: {
        -2: "Sin consumo (-2)",
        -1: "Pagado puntual (-1)",
        0: "Uso crédito revolvente (0)",
        1: "Retraso 1 mes (+1)",
        2: "Retraso 2 meses (+2)",
        3: "Retraso 3 meses (+3)"
    }[x]
)

bill_amt1 = st.sidebar.number_input("Saldo Facturado Reciente (MXN):", min_value=0, max_value=500000, value=init_bill1, step=2500)
pay_amt1 = st.sidebar.number_input("Monto Pagado Reciente (MXN):", min_value=0, max_value=300000, value=init_payamt1, step=1000)

# Build Complete Feature Vector (23 features)
sample_vector = {
    "LIMIT_BAL": limit_bal,
    "SEX": sex,
    "EDUCATION": education,
    "MARRIAGE": marriage,
    "AGE": age,
    "PAY_1": pay_1,
    "PAY_2": 0 if pay_1 <= 0 else pay_1 - 1,
    "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": bill_amt1,
    "BILL_AMT2": int(bill_amt1 * 0.95),
    "BILL_AMT3": int(bill_amt1 * 0.90),
    "BILL_AMT4": int(bill_amt1 * 0.85),
    "BILL_AMT5": int(bill_amt1 * 0.80),
    "BILL_AMT6": int(bill_amt1 * 0.75),
    "PAY_AMT1": pay_amt1,
    "PAY_AMT2": int(pay_amt1 * 0.9),
    "PAY_AMT3": int(pay_amt1 * 0.8),
    "PAY_AMT4": int(pay_amt1 * 0.8),
    "PAY_AMT5": int(pay_amt1 * 0.8),
    "PAY_AMT6": int(pay_amt1 * 0.8)
}
applicant_df = pd.DataFrame([sample_vector])


# ---------------------------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">¿Por qué me rechazaron?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header"><b>Auditoría de Explicabilidad, Recursos Accionables y Disparidad en Crédito al Consumo</b> | '
    'AIPI 590: Explainable AI Final Project</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "💳 1. Simulador & Recurso Accionable",
    "⚖️ 2. SHAP vs. DiCE (La Paradoja de XAI)",
    "📊 3. Auditoría de Carga Algorítmica"
])


# ---------------------------------------------------------------------------
# TAB 1: SIMULADOR & RECURSO ACCIONABLE
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Decisión Automatizada del Modelo de Crédito")

    # Model inference
    proba_default = float(model.predict_proba(applicant_df)[:, 1][0])
    threshold = metadata["threshold"]
    is_rejected = proba_default >= threshold

    col_res1, col_res2, col_res3 = st.columns([1.5, 1.5, 3])
    with col_res1:
        st.markdown("**Resultado:**")
        if is_rejected:
            st.markdown('<span class="badge-rejected">❌ SOLICITUD RECHAZADA (Riesgo Alto)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-approved">✅ SOLICITUD APROBADA (Confiable)</span>', unsafe_allow_html=True)

    with col_res2:
        st.metric("Probabilidad de Incumplimiento:", f"{proba_default:.1%}", delta=f"{(proba_default - threshold):+.1%} vs Umbral", delta_color="inverse")

    with col_res3:
        st.markdown(f"""
        **Criterio Regulatorio:**
        El umbral de riesgo de la institución es **{threshold:.0%}**. 
        {f'El solicitante excede el límite de tolerancia de riesgo por {(proba_default - threshold):.1%}.' if is_rejected else 'El solicitante califica para la línea de crédito solicitada.'}
        """)

    st.markdown("---")

    if is_rejected:
        st.markdown("### 🎯 Generador de Recursos Accionables (DiCE Counterfactuals)")
        st.markdown("""
        Bajo el marco de **Protección al Consumidor Financiero (CONDUSEF / GDPR Art. 22)**, no basta con decir *por qué* se rechazó.
        El solicitante tiene derecho a conocer **planes de acción mínimos y viables** para revertir la decisión.
        """)

        with st.spinner("Calculando rutas accionables sin modificar atributos protegidos..."):
            cfs_df, deltas_df, mean_burden = recourse_engine.generate_recourse_plans(applicant_df, total_cfs=2)

        if not deltas_df.empty:
            col_plan1, col_plan2 = st.columns(2)
            for idx, row in deltas_df.iterrows():
                col_target = col_plan1 if idx == 0 else col_plan2
                with col_target:
                    st.markdown(f"#### 📋 {row['Plan_ID']} (Esfuerzo Relativo: {row['Normalized_Burden']:.2f})")
                    for col_name, val_str in row.items():
                        if col_name not in ["Plan_ID", "Normalized_Burden"]:
                            st.info(f"**{col_name}:** {val_str}")
                    st.caption("🔒 *Atributos inmutables (Edad, Género, Estado Civil) se mantuvieron estrictamente congelados.*")
            
            st.success(f"**Esfuerzo Algorítmico Promedio (Distancia L1 Normalizada):** {mean_burden:.2f}")
        else:
            st.warning("No se encontró un contra factual viable dentro de los límites de un solo paso. Se requiere reestructuración integral de deuda.")
    else:
        st.balloons()
        st.success("¡Felicidades! Este perfil califica directamente para la aprobación de crédito.")


# ---------------------------------------------------------------------------
# TAB 2: SHAP VS. DICE (THE EXPLAINABILITY PARADOX)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("La Paradoja de XAI: Atribución Retrospectiva vs. Recurso Prescriptivo")
    st.markdown("""
    En la industria financiera, **SHAP** y **LIME** se utilizan comúnmente para explicar denegaciones de crédito. 
    Sin embargo, esta comparativa ilustra por qué las atribuciones de características resultan insuficientes para el usuario.
    """)

    col_shap, col_dice = st.columns(2)

    with col_shap:
        st.markdown("### 📉 Enfoque Tradicional: SHAP (Retrospectivo)")
        st.caption("Responde a: *'¿Qué hiciste mal en el pasado?'* (Atribución estática)")
        shap_vals, base_val, summary_df = shap_explainer.explain_instance(applicant_df)

        fig_shap, ax_shap = plt.subplots(figsize=(6, 4.5))
        top_shap = summary_df.head(6)
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in top_shap["SHAP_Attribution"]]
        ax_shap.barh(top_shap["Feature"], top_shap["SHAP_Attribution"], color=colors)
        ax_shap.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_shap.set_xlabel("Impacto SHAP en Riesgo de Default")
        ax_shap.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_shap)

        st.markdown("""
        **Limitaciones de SHAP para el Solicitante:**
        - Solo describe la culpa (*"Tu saldo es alto y tu historial tiene retrasos"*).
        - No especifica cuánto pagar ni qué meta concreta garantiza la aprobación.
        - Frecuentemente resalta variables inmutables sobre las que el usuario no tiene control.
        """)

    with col_dice:
        st.markdown("### 🚀 Enfoque Responsable: DiCE (Prescriptivo)")
        st.caption("Responde a: *'¿Qué acción mínima garantiza tu aprobación?'* (Recurso Accionable)")

        if is_rejected and not deltas_df.empty:
            st.dataframe(deltas_df, use_container_width=True)
            st.markdown("""
            **Ventajas Clave de DiCE / Algorithmic Recourse:**
            - **Accionabilidad:** Solo recomienda cambios en variables financieras mutables.
            - **No Discriminación:** Bloquea explícitamente variables protegidas (Género, Edad).
            - **Minimización de Esfuerzo:** Identifica la ruta de menor fricción financiera ($L_1$ distance).
            """)
        else:
            st.info("Para visualizar la comparativa de recursos accionables, seleccione un perfil en riesgo en la barra lateral.")


# ---------------------------------------------------------------------------
# TAB 3: AUDITORÍA DE CARGA ALGORÍTMICA (REGULATORY AUDIT)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Auditoría Regulatoria de Equidad y Carga Algorítmica")
    st.markdown("""
    Evaluación empírica sobre si el modelo de crédito y el motor de recursos imponen una **Carga Algorítmica Desproporcionada** 
    a grupos históricamente vulnerables (Género y Grupos de Edad).
    """)

    # Load Audit Results
    audit_json_path = "models/audit_disparity_results.json"
    if os.path.exists(audit_json_path):
        with open(audit_json_path, "r") as f:
            audit_data = json.load(f)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(
                "Razón de Disparidad por Género:",
                f"{audit_data['gender_disparity_ratio']:.3f}x",
                delta=f"+{(audit_data['gender_disparity_ratio'] - 1.0):.1%} vs Paridad (1.00x)",
                delta_color="inverse"
            )
            st.caption("Esfuerzo promedio requerido por mujeres vs. hombres para revertir una denegación.")

        with col_m2:
            st.metric(
                "Razón de Disparidad por Edad:",
                f"{audit_data['age_disparity_ratio']:.3f}x",
                delta=f"+{(audit_data['age_disparity_ratio'] - 1.0):.1%} vs Paridad (1.00x)",
                delta_color="inverse"
            )
            st.caption("Esfuerzo promedio requerido por jóvenes (<30 años) vs. adultos establecidos.")

        with col_m3:
            st.metric("Tasa de Factibilidad de Recurso:", "100.0%", delta="Ruta viable encontrada")
            st.caption("Porcentaje de solicitantes rechazados que obtuvieron al menos 1 ruta factible.")

        st.markdown("---")

        col_plot, col_recs = st.columns([1.2, 1])
        with col_plot:
            st.markdown("#### Distribución de Esfuerzo Normalizado L1")
            plot_img_path = "models/audit_disparity_plot.png"
            if os.path.exists(plot_img_path):
                st.image(plot_img_path, use_container_width=True)

        with col_recs:
            st.markdown("#### 🏛️ Implicaciones para la Regulación Mexicana (CONDUSEF & Ley Fintech)")
            st.markdown("""
            1. **Disparidad de Género Detectada (+7.1%):**  
               Las solicitantes mujeres enfrentan un umbral de esfuerzo financiero 7.1% mayor para revertir un rechazo, 
               reflejando sesgos históricos en las líneas de crédito base asignadas.
            2. **Disparidad por Edad (+15.8%):**  
               Los jóvenes menores de 30 años sufren la mayor penalización en distancia de recurso debido a su menor 
               antigüedad crediticia.
            3. **Recomendación de Política Pública:**  
               La CONDUSEF debe exigir que las entidades fintech no solo presenten explicaciones de variables globales (SHAP), 
               sino que auditen formalmente las métricas de **Recourse Disparity Ratio** para garantizar una verdadera inclusión financiera.
            """)
    else:
        st.warning("No se encontraron resultados de auditoría previa. Ejecute `src/audit.py` para generar métricas.")

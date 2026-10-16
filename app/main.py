"""
=============================================================================
APPLICATION: main.py (Streamlit Web Dashboard)
PROJECT: AIPI 590 - Project 1: ¿Por qué me rechazaron?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Interactive portfolio artifact demonstrating Explainable AI, Algorithmic
    Recourse (DiCE), and Multi-Dimensional Fairness Auditing with Statistical
    Significance Testing for credit scoring underwriting.
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

# Relative imports with robust base directory resolution
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from src.data_loader import (
    CATEGORICAL_LABELS,
    DEFAULT_DATA_PATH
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
        font-size: 1.05rem;
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
    model_path = os.path.join(BASE_DIR, "models", "credit_model.joblib")
    meta_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    shap_explainer = CreditSHAPExplainer(model_path=model_path)
    recourse_engine = CreditRecourseEngine(
        model_path=model_path,
        data_path=os.path.join(BASE_DIR, "data", "credit_card_clean.csv"),
        sample_size=1500
    )
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
        -2: "Sin consumo / saldo cero (-2)",
        -1: "Pagado puntual íntegro (-1)",
        0: "Uso crédito revolvente c/pago mínimo (0)",
        1: "Retraso 1 mes (+1)",
        2: "Retraso 2 meses (+2)",
        3: "Retraso 3 meses (+3)"
    }[x]
)
st.sidebar.caption("ℹ️ *Nota de Dominio:* En la taxonomía UCI Credit, -2 y -1 representan cumplimiento superior o saldo liquidado, mientras que 0 indica financiamiento revolvente y valores positivos reflejan mora.")

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
        st.metric(
            "Probabilidad de Incumplimiento:",
            f"{proba_default:.1%}",
            delta=f"{(proba_default - threshold):+.1%} vs Umbral",
            delta_color="inverse"
        )

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

        with st.spinner("Calculando rutas accionables bajo restricciones causales estrictas..."):
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
                    st.caption("🔒 *Atributos inmutables (Edad, Género, Educación, Estado Civil), historial previo y límites bancarios se mantuvieron estrictamente congelados.*")
            
            st.success(f"**Esfuerzo Algorítmico Promedio (Distancia L1 Condicional Normalizada):** {mean_burden:.2f}")
        else:
            st.warning("No se encontró un contra factual viable dentro de los límites de un solo paso. Se requiere reestructuración integral de deuda a mediano plazo.")
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
            - **Accionabilidad Estricta:** Solo modifica palancas financieras presentes (`PAY_1`, `BILL_AMT1`, `PAY_AMT1`).
            - **No Discriminación:** Bloquea explícitamente variables protegidas (Género, Edad, Educación, Estado Civil).
            - **Restricción Causal y Temporal:** No altera el historial crediticio pasado (`PAY_2..6`) ni límites fijados por el banco (`LIMIT_BAL`).
            - **Restricciones Monótonas:** No permite empeorar el estatus de pago ni pagar más que el saldo facturado.
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
    Evaluación empírica en el **conjunto de prueba fuera de muestra (hold-out test set)** para evaluar si el modelo de crédito 
    impone una **Carga Algorítmica Desproporcionada** a través de 4 dimensiones demográficas (Género, Edad, Educación y Estado Civil).
    """)

    audit_json_path = os.path.join(BASE_DIR, "models", "audit_disparity_results.json")
    if os.path.exists(audit_json_path):
        with open(audit_json_path, "r") as f:
            audit_data = json.load(f)

        comps = audit_data["comparisons"]
        subgroups = audit_data["subgroups"]
        overall = audit_data["overall"]

        # 4 Key Demographic Dimensions Cards
        st.markdown("#### 1. Razones de Disparidad por Dimensión Demográfica")
        col_g, col_a, col_e, col_m = st.columns(4)

        with col_g:
            g_comp = comps["gender"]
            ratio_g = g_comp["disparity_ratio"]
            st.metric(
                "Género (Mujer / Hombre):",
                f"{ratio_g:.3f}x",
                delta=f"{(ratio_g - 1.0):+.1%} vs Paridad (1.00x)",
                delta_color="inverse" if g_comp["is_burden_statistically_significant"] else "off"
            )
            st.caption(f"Welch t-test: p = {g_comp['welch_t_pvalue']:.3f} (p_adj = {g_comp['welch_t_pvalue_holm_adjusted']:.3f}) | {'Significativo' if g_comp['is_burden_statistically_significant'] else 'Sin diferencia significativa'}")

        with col_a:
            a_comp = comps["age"]
            ratio_a = a_comp["disparity_ratio"]
            st.metric(
                "Edad (Jóvenes / Adultos 30+):",
                f"{ratio_a:.3f}x",
                delta=f"{(ratio_a - 1.0):+.1%} vs Paridad (1.00x)",
                delta_color="inverse" if a_comp["is_burden_statistically_significant"] else "off"
            )
            st.caption(f"Welch t-test: p = {a_comp['welch_t_pvalue']:.3f} (p_adj = {a_comp['welch_t_pvalue_holm_adjusted']:.3f}) | {'Significativo' if a_comp['is_burden_statistically_significant'] else 'Sin diferencia significativa'}")

        with col_e:
            e_comp = comps["education"]
            ratio_e = e_comp["disparity_ratio"]
            st.metric(
                "Educación (Media / Superior):",
                f"{ratio_e:.3f}x",
                delta=f"{(ratio_e - 1.0):+.1%} vs Paridad (1.00x)",
                delta_color="inverse" if e_comp["is_burden_statistically_significant"] else "off"
            )
            st.caption(f"Welch t-test: p = {e_comp['welch_t_pvalue']:.3f} (p_adj = {e_comp['welch_t_pvalue_holm_adjusted']:.3f}) | {'Significativo' if e_comp['is_burden_statistically_significant'] else 'Sin diferencia significativa'}")

        with col_m:
            m_comp = comps["marriage"]
            ratio_m = m_comp["disparity_ratio"]
            st.metric(
                "Estado Civil (Soltero / Casado):",
                f"{ratio_m:.3f}x",
                delta=f"{(ratio_m - 1.0):+.1%} vs Paridad (1.00x)",
                delta_color="inverse" if m_comp["is_burden_statistically_significant"] else "off"
            )
            st.caption(f"Welch t-test: p = {m_comp['welch_t_pvalue']:.3f} (p_adj = {m_comp['welch_t_pvalue_holm_adjusted']:.3f}) | {'Significativo' if m_comp['is_burden_statistically_significant'] else 'Sin diferencia significativa'}")

        st.info(
            "⚠️ **Aviso Metodológico sobre Sesgo de Supervivencia:** Una razón de carga condicional inferior a 1.00x "
            "**no** debe interpretarse como menor esfuerzo si va acompañada de una menor tasa de factibilidad. "
            "El esfuerzo se calcula exclusivamente sobre solicitantes que logran obtener un plan viable dentro de las restricciones; "
            "los perfiles en mayor estrés financiero son descartados como inviables, lo que puede subestimar la carga media de dichos grupos."
        )

        st.markdown("---")

        # Overall Metrics Cards
        col_ov1, col_ov2, col_ov3 = st.columns(3)
        with col_ov1:
            st.metric("Tasa de Factibilidad Global:", f"{overall['feasibility_rate_pct']:.1f}%", delta="Rutas viables encontradas")
            st.caption(f"{overall['successful_total']} de {overall['evaluated_total']} solicitudes evaluadas en conjunto de prueba.")
        with col_ov2:
            st.metric("Carga Algorítmica Condicional Media:", f"{overall['mean_burden']:.2f}", delta=f"IC 95%: [{overall['ci_95_lower']:.2f}, {overall['ci_95_upper']:.2f}]")
            st.caption("Distancia L1 normalizada en desviaciones estándar de la población.")
        with col_ov3:
            st.metric("Población Denegada en Test:", f"{audit_data['metadata']['model_rejected_sample_size']:,}", delta=f"De {audit_data['metadata']['test_sample_size']:,} casos evaluados")
            st.caption(f"Solicitudes con P(default) >= {audit_data['metadata']['threshold']:.0%}.")

        st.markdown("---")

        col_plot, col_recs = st.columns([1.3, 1])
        with col_plot:
            st.markdown("#### Distribución de Esfuerzo Normalizado L1 (Con Intervalos de Confianza 95%)")
            plot_img_path = os.path.join(BASE_DIR, "models", "audit_disparity_plot.png")
            if os.path.exists(plot_img_path):
                st.image(plot_img_path, use_container_width=True)

        with col_recs:
            st.markdown("#### 🏛️ Conclusiones y Recomendaciones Regulatorias (CONDUSEF & Ley Fintech)")
            
            sub_high = subgroups.get("Higher Ed (Grad/Univ)", {})
            sub_sec = subgroups.get("Secondary / Other", {})
            sub_yng = subgroups.get("Young (< 30)", {})
            sub_old = subgroups.get("Adultos (≥ 30)", subgroups.get("Older (>= 30)", {}))
            
            st.markdown(f"""
            1. **Dimensión de Género ({g_comp['ratio_label']}: {g_comp['disparity_ratio']:.3f}x):**  
               {g_comp['burden_conclusion']}
            2. **Dimensión de Edad ({a_comp['ratio_label']}: {a_comp['disparity_ratio']:.3f}x):**  
               {a_comp['burden_conclusion']}  
               *Factibilidad:* {sub_yng.get('feasibility_rate_pct', 0.0):.1f}% en jóvenes vs. {sub_old.get('feasibility_rate_pct', 0.0):.1f}% en adultos de 30+ ({a_comp['feasibility_conclusion']}).
            3. **Dimensión Educativa ({e_comp['ratio_label']}: {e_comp['disparity_ratio']:.3f}x):**  
               {e_comp['burden_conclusion']}  
               *Factibilidad:* {sub_high.get('feasibility_rate_pct', 0.0):.1f}% en educación superior vs. {sub_sec.get('feasibility_rate_pct', 0.0):.1f}% en educación media ({e_comp['feasibility_conclusion']}).
            4. **Dimensión de Estado Civil ({m_comp['ratio_label']}: {m_comp['disparity_ratio']:.3f}x):**  
               {m_comp['burden_conclusion']}
            5. **Control de Sesgo de Supervivencia y Corrección Múltiple:**  
               La carga condicional solo evalúa casos con solución factible ({overall['feasibility_rate_pct']:.1f}% global). Por ello, se audita simultáneamente la factibilidad mediante la prueba exacta de Fisher y se reportan correcciones Holm-Bonferroni para controlar la tasa de error por familia (tanto a nivel dominio de 4 pruebas como global de 8 pruebas).
            """)

        # Statistical Testing Detail Expander
        with st.expander("🔬 Ver Detalles de Pruebas de Hipótesis y Corrección Holm-Bonferroni"):
            st.markdown("### Tabla Comparativa de Pruebas Estadísticas")
            detail_records = []
            for dim_key, c_info in comps.items():
                detail_records.append({
                    "Dimensión": c_info["dimension"],
                    "Comparación": f"{c_info['group_a']} vs {c_info['group_b']}",
                    "Razón Carga": f"{c_info['disparity_ratio']:.3f}x",
                    "Welch p (raw)": f"{c_info['welch_t_pvalue']:.4f}",
                    "Welch p (Holm 4-tests)": f"{c_info['welch_t_pvalue_holm_adjusted']:.4f}",
                    "Welch p (Holm global 8-tests)": f"{c_info.get('welch_t_pvalue_holm_global_8tests', c_info['welch_t_pvalue_holm_adjusted']):.4f}",
                    "Brecha Factibilidad": f"{c_info['feasibility_gap_pct']:+.1f}%",
                    "Fisher p (raw)": f"{c_info['fisher_feasibility_pvalue']:.4f}",
                    "Fisher p (Holm 4-tests)": f"{c_info['fisher_pvalue_holm_adjusted']:.4f}",
                    "Fisher p (Holm global 8-tests)": f"{c_info.get('fisher_pvalue_holm_global_8tests', c_info['fisher_pvalue_holm_adjusted']):.4f}",
                    "Significativo (p_adj < 0.05)": "Sí (p_adj < 0.05)" if (c_info["is_burden_statistically_significant"] or c_info["is_feasibility_statistically_significant"]) else "No (p >= 0.05)"
                })
            st.dataframe(pd.DataFrame(detail_records), use_container_width=True)

    else:
        st.warning("No se encontraron resultados de auditoría previa. Ejecute `python src/audit.py` para generar métricas.")

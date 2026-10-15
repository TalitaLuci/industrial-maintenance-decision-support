"""Dashboard Streamlit para explorar previsões e priorização demonstrativa."""
from pathlib import Path
import pandas as pd
import streamlit as st
from src.decision import run_ranking

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
PRED_PATH = REPORTS / "test_predictions.csv"

st.set_page_config(page_title="Maintenance Decision Support", layout="wide")
st.title("Manutenção preditiva e priorização multicritério")
st.caption("Projeto de portfólio com NASA C-MAPSS FD001 — resultados demonstrativos.")

st.warning(
    "Dados de motores simulados. Criticidade e índices econômicos são hipotéticos. "
    "Este dashboard não fornece recomendações operacionais reais."
)

if not PRED_PATH.exists():
    st.info(
        "As previsões ainda não existem. Baixe os arquivos C-MAPSS FD001, coloque-os "
        "em data/raw/ e execute `python -m src.modeling`."
    )
    st.stop()

pred = pd.read_csv(PRED_PATH)
metrics_path = REPORTS / "test_metrics.json"
if metrics_path.exists():
    import json
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (ciclos)", f"{metrics['mae']:.2f}")
    c2.metric("RMSE (ciclos)", f"{metrics['rmse']:.2f}")
    c3.metric("Escore C-MAPSS", f"{metrics['cmapss_score']:.2f}")

st.subheader("Distribuição de RUL")
st.write("Comparação entre a RUL verdadeira e a prevista na avaliação retrospectiva do teste.")
st.scatter_chart(pred, x="rul_true", y="rul_pred")
st.caption("A RUL verdadeira é usada apenas para avaliar o modelo; não é uma entrada para a priorização.")

st.subheader("Priorização demonstrativa")
profile_label = st.selectbox(
    "Perfil de decisão",
    ["Equilibrado", "Segurança primeiro", "Custo primeiro"],
)
profile_map = {
    "Equilibrado": "balanced",
    "Segurança primeiro": "safety_first",
    "Custo primeiro": "cost_first",
}
ranked, meta = run_ranking(pred, profile_map[profile_label])
st.write("Pesos AHP do perfil selecionado")
st.json(meta["weights"])
st.write(f"Razão de consistência AHP: {meta['consistency_ratio']:.3f}")
if not meta["consistency_acceptable_reference"]:
    st.warning("A razão de consistência excede 0,10; revise as comparações par a par.")
st.dataframe(
    ranked[[
        "unit_nr", "rul_pred", "urgency", "operational_criticality",
        "intervention_cost", "downtime_impact", "topsis_score", "rank"
    ]].rename(columns={
        "unit_nr": "Motor",
        "rul_pred": "RUL prevista (ciclos)",
        "urgency": "Urgência (transformada)",
        "operational_criticality": "Criticidade hipotética",
        "intervention_cost": "Índice de custo hipotético",
        "downtime_impact": "Impacto hipotético",
        "topsis_score": "Pontuação TOPSIS",
        "rank": "Posição",
    }),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Os índices de criticidade, custo e impacto são sintéticos e criados por regras "
    "determinísticas de demonstração. Não representam informações reais sobre cada motor."
)

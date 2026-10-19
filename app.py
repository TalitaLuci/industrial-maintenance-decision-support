"""Dashboard Streamlit para explorar previsões e priorização demonstrativa."""
from pathlib import Path
import pandas as pd
import streamlit as st
import json
from src.decision import run_ranking, URGENCY_HORIZON
from src.economics import (
    wasted_life_cost, break_even_probability, compare_cost_scenarios, describe_break_even,
)

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
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    base = metrics.get("baseline", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (ciclos)", f"{metrics['mae']:.1f}",
              delta=f"baseline {base['mae']:.1f}" if base else None, delta_color="off")
    c2.metric("RMSE (ciclos)", f"{metrics['rmse']:.1f}",
              delta=f"baseline {base['rmse']:.1f}" if base else None, delta_color="off")
    c3.metric("Escore C-MAPSS", f"{metrics['cmapss_score']:.0f}",
              delta=f"baseline {base['cmapss_score']:.0f}" if base else None, delta_color="off")
    if metrics.get("rul_cap"):
        st.caption(f"Modelo treinado com RUL limitada a {metrics['rul_cap']} ciclos (decisão do notebook 01).")

st.subheader("RUL real vs. prevista (teste oficial)")
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
        "intervention_cost", "failure_impact", "topsis_score", "rank"
    ]].rename(columns={
        "unit_nr": "Motor",
        "rul_pred": "RUL prevista (ciclos)",
        "urgency": "Urgência (transformada)",
        "operational_criticality": "Criticidade hipotética",
        "intervention_cost": "Índice de custo hipotético",
        "failure_impact": "Impacto da falha (hipotético)",
        "topsis_score": "Pontuação TOPSIS",
        "rank": "Posição",
    }),
    width="stretch",
    hide_index=True,
)
st.caption(
    "Os índices de criticidade, custo e impacto da falha são sintéticos (sorteio com semente fixa). "
    "Não representam informações reais sobre cada motor."
)

st.subheader("Cenário econômico ilustrativo: intervir agora ou adiar?")
st.write(
    "Valores monetários hipotéticos. Intervir agora desperdiça parte da vida útil restante; "
    "adiar aceita a chance de falha antes da intervenção."
)
motor = st.selectbox("Motor", ranked["unit_nr"].tolist(), index=0)
rul_motor = float(ranked.loc[ranked["unit_nr"] == motor, "rul_pred"].iloc[0])
col1, col2, col3 = st.columns(3)
falha = col1.slider("Custo da falha (u.m.)", 500, 5000, 2000, step=250)
valor = col2.slider("Valor do componente (u.m.)", 100, 1000, 400, step=50)
p = col3.slider("Chance de falhar antes de intervir", 0.0, 0.5, 0.10, step=0.01)
desperdicio = wasted_life_cost(rul_motor, valor, URGENCY_HORIZON)
p_star = break_even_probability(desperdicio, falha)
st.write(f"RUL prevista do motor {motor}: **{rul_motor:.0f} ciclos**")
st.info(describe_break_even(p_star))
st.dataframe(
    compare_cost_scenarios(100.0, 50.0, float(falha), p, desperdicio).round(1),
    width="stretch", hide_index=True,
)
st.caption("Intervenção = 100 u.m. e parada planejada = 50 u.m. (fixos, nos dois cenários).")

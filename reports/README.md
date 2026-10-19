# Resultados gerados

Arquivos produzidos pelo código (pequenos e versionados para que o dashboard funcione logo após clonar o repositório):

| Arquivo | Origem | Conteúdo |
|---|---|---|
| `cv_metrics.csv` | `python -m src.modeling` | Métricas de validação cruzada por motor (5 partições), baseline e modelo, variante final (teto de RUL = 125) |
| `cap_comparison.csv` | notebook 01 | Mesma validação para as variantes sem teto e com teto |
| `test_predictions.csv` | `python -m src.modeling` | RUL real e prevista dos 100 motores do teste oficial (último ciclo observado) |
| `test_metrics.json` | `python -m src.modeling` | MAE, RMSE, escore C-MAPSS e viés no teste oficial, com o baseline |
| `ranking_*.csv` | notebook 02 | Ranking TOPSIS por perfil de decisor (critérios hipotéticos) |
| `figures/*.png` | notebooks 01 e 02 | Gráficos usados no README |

O teste oficial é reservado para a avaliação final: a variante e os hiperparâmetros são escolhidos na validação cruzada, não com `test_metrics.json`. Ressalva: o teste de um modelo inicial foi inspecionado de forma diagnóstica durante o desenvolvimento (veja o README), então o resultado não é uma estimativa totalmente independente.

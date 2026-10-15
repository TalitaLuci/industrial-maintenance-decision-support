# Industrial Maintenance Decision Support
## Manutenção preditiva de turbofans e priorização multicritério de intervenções

Projeto demonstrativo de Ciência de Dados com o subconjunto **NASA C-MAPSS FD001**. O objetivo é estimar a vida útil restante (Remaining Useful Life — RUL) de motores turbofan e combinar essa estimativa com critérios hipotéticos de manutenção por meio de AHP e TOPSIS.

> **Limitação essencial:** o conjunto de dados é simulado. Criticidade e valores econômicos usados neste projeto são cenários demonstrativos, não dados reais de operação. Os rankings não devem ser usados para decisões reais de manutenção.

## Perguntas do projeto

1. É possível estimar a RUL de motores a partir de dados de sensores?
2. Como o modelo se comporta em motores não vistos durante o treinamento?
3. Como um ranking multicritério muda quando variam urgência, criticidade e custos?
4. O ranking permanece estável quando os pesos e as premissas mudam?

## Estrutura

```text
industrial-maintenance-decision-support/
├── app.py
├── requirements.txt
├── data/                 # arquivos C-MAPSS locais (não incluídos)
├── notebooks/            # roteiro de análise exploratória
├── reports/              # saídas geradas
├── src/
│   ├── data_processing.py
│   ├── features.py
│   ├── metrics.py
│   ├── modeling.py
│   ├── ahp.py
│   ├── topsis.py
│   ├── decision.py
│   └── economics.py
└── tests/
```

## Dados

Baixe o conjunto C-MAPSS na fonte oficial da NASA e coloque os arquivos abaixo em `data/raw/`:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

A estrutura de colunas é composta por identificador do motor, ciclo, três configurações operacionais e 21 sensores. O FD001 representa um regime operacional e um modo de falha simulados.

Os arquivos de dados não estão incluídos neste repositório/ZIP. Consulte a página de dados da NASA: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

## Instalação

Recomendado: Python 3.10 ou 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Uso rápido

### 1. Auditar dados e treinar/avaliar o modelo

```bash
python -m src.modeling
```

Esse comando lê os arquivos locais, cria RUL de treino, gera atributos de janela retrospectiva, usa GroupKFold agrupado por motor para avaliação de desenvolvimento e avalia o modelo final no último ciclo de cada motor do conjunto oficial de teste. Salva previsões e métricas em `reports/`.

### 2. Executar o dashboard

```bash
streamlit run app.py
```

O dashboard carrega as previsões e cria uma priorização demonstrativa. Se as saídas ainda não existirem, ele mostra instruções.

### 3. Executar testes

```bash
pytest -q
```

## Metodologia e cuidados

- A divisão de desenvolvimento agrupa por motor (`GroupKFold`) para evitar que ciclos do mesmo motor apareçam em treino e validação na mesma divisão.
- O conjunto oficial de teste só deve ser usado depois da seleção do modelo. Não use os resultados do teste para ajustar hiperparâmetros.
- A avaliação oficial do FD001 é feita no último ciclo observado de cada motor de teste, comparando com `RUL_FD001`.
- As features de janela usam apenas a observação atual e ciclos anteriores do mesmo motor.
- O exemplo de modelo usa `HistGradientBoostingRegressor`. Não presume que restrições monotônicas sejam sempre corretas.
- A implementação AHP inclui razão de consistência. TOPSIS orienta cada critério como benefício ou custo.
- Criticidade e custos são explicitamente hipotéticos. O script de decisão fornece regras demonstrativas reproduzíveis e não representa uma base operacional.
- Resultados reais devem ser inseridos no README após executar o projeto. Não reporte métricas que ainda não foram calculadas.

## Métricas

- MAE: erro absoluto médio em ciclos.
- RMSE: raiz do erro quadrático médio em ciclos.
- Escore assimétrico C-MAPSS: penaliza mais a superestimação da RUL. Convenção implementada: `erro = previsão - valor_real`; erro positivo significa superestimação.

## Próximas melhorias

- Comparar baseline simples e diferentes algoritmos com protocolo idêntico.
- Avaliar modelos com restrições monotônicas somente quando houver justificativa física/documental.
- Investigar importância de atributos e erros por faixa de RUL.
- Adicionar análise de sensibilidade e cenários econômicos com premissas explicitadas.
- Avaliar incerteza com método apropriado antes de apresentar probabilidades calibradas.

## Aviso

Este repositório é um projeto de portfólio educacional. Não é um sistema certificado, não oferece recomendação operacional e não substitui análise de engenharia, procedimentos de segurança ou validação com dados reais.

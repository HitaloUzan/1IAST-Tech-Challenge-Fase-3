# Tech Challenge - Fase 3: Predicao da Alfabetizacao Infantil e Busca Ativa

## 1. Contexto e Objetivos de Negocio

A alfabetizacao na idade certa e um dos principais desafios da educacao basica publica no Brasil. Diante da necessidade de intervencoes pedagogicas direcionadas e alocacao eficiente de recursos publicos, este projeto desenvolve uma pipeline completa de Machine Learning integrada a arquitetura GCP (BigQuery - Camada Gold).

O objetivo e predizer a probabilidade de um aluno estar alfabetizado ao final do ciclo de alfabetizacao, funcionando como uma ferramenta de apoio a decisao para redes municipais e estaduais na implementacao de politicas de Busca Ativa e Reforco Escolar.

---

## 2. Arquitetura da Pipeline e Estrutura do Projeto

A solucao possui arquitetura modularizada, garantindo reprodutibilidade e aderencia as melhores praticas de Engenharia de Machine Learning:

```text
├── config.py                 # Credenciais GCP e constantes globais
├── requirements.txt          # Dependencias do projeto
├── README.md                 # Documentacao oficial
├── notebooks/                # Prototipagem e analises exploratorias
├── reports/
│   ├── images/               # Graficos da EDA, Feature Importance e SHAP
│   └── model_results.csv     # Tabela comparativa de metricas
├── models/                   # Artefatos binarios (.joblib) dos modelos
└── src/
    ├── gold/
    │   └── build_gold_ml.py  # Automacao de criacao da camada Gold no BigQuery
    └── ml/
        ├── exploratory_analysis.py # EDA e geracao de graficos estatisticos
        ├── preprocessing.py        # Split 80/20, Imputer, Scaler e Sampler
        ├── train.py                # Treinamento comparativo dos modelos
        ├── evaluate.py             # Avaliacao tecnica e Threshold Tuning (0.55)
        ├── predict.py              # Motor de inferencia e classificacao operacional
        └── explain.py              # Explicabilidade global e local (SHAP / Gain)

```

### Protecao contra Data Leakage e Pre-Processamento

* **Filtro de Escopo:** Foco exclusivo em escolas das Redes Publicas (Municipais e Estaduais).
* **Split Estratificado (80/20):** Separacao de treino e teste mantendo a proporcao real da variavel alvo em uma base de teste de 670.928 registros.
* **Isolamento de Transformadores:** As etapas de imputacao (`SimpleImputer`) e padronizacao (`StandardScaler`) realizam o `.fit()` estritamente na base de treino.
* **Balanceamento Restrito ao Treino:** Aplicacao do `RandomUnderSampler` (50/50) exclusivo no treino para nao contaminar a avaliacao no conjunto de teste.

## 3. Analise Exploratoria e Importancia das Variaveis

A analise da camada Gold revela a forte influencia de indicadores historicos e do contexto socioeconomico municipal no desempenho dos alunos.

### Tabela de Contribuicao Relativa das Variaveis (Feature Importance - XGBoost Gain)

| **Variavel**                       | **Descricao Pratica**                              | **Contribuicao Relativa (%)** |
| ---------------------------------- | -------------------------------------------------- | ----------------------------- |
| **`taxa_alfabetizacao_municipio`** | Historico/taxa de alfabetizacao do municipio       | **80,4%**                     |
| **`rede_Estadual`**                | Aluno estuda em Escola Estadual (Sim/Nao)          | **6,8%**                      |
| **`proporcao_adequado_avancado`**  | % de alunos nos niveis avancados de aprendizado    | **2,8%**                      |
| **`peso_aluno`**                   | Peso amostral do aluno na avaliacao do INEP        | **2,2%**                      |
| **`media_portugues_municipio`**    | Desempenho medio do municipio em Lingua Portuguesa | **2,1%**                      |
| **`proporcao_basico`**             | % de alunos no nivel basico de aprendizado         | **2,0%**                      |
| **`inse_municipio`**               | Indicador do Nivel Socioeconomico do Municipio     | **1,9%**                      |
| **`proporcao_abaixo_basico`**      | % de alunos na camada mais critica de aprendizado  | **1,6%**                      |
| **`rede_Municipal`**               | Aluno estuda em Escola Municipal (Sim/Nao)         | **0,2%**                      |

### Explicabilidade do Modelo (SHAP Values)

O uso de SHAP (SHapley Additive exPlanations) valida a logica do modelo:

1. **Impacto Direcional (Summary Plot):** Maiores taxas historicas de alfabetizacao e maior nivel socioeconomico (INSE) deslocam as probabilidades positivamente.
2. **Diagnostico Individual (Waterfall):** Explicita quais variaveis atuaram como fatores de risco ou protecao para um aluno especifico, permitindo intervencoes direcionadas.

## 4. Comparacao de Modelos e Resultados Tecnicos

A avaliacao tecnica comparou tres algoritmos no conjunto de teste nao balanceado contendo 670.928 registros:

### Tabela Comparativa de Desempenho (Threshold Padrao = 0,50)

| **Modelo**              | **Acuracia Baseline** | **ROC-AUC** | **F1-Macro** | **Recall (Nao Alfab. - Risco)** | **Recall (Alfabetizado)** |
| ----------------------- | --------------------- | ----------- | ------------ | ------------------------------- | ------------------------- |
| **Regressao Logistica** | 61,96%                | 0.6723      | 0.6154       | 63,01%                          | 61,24%                    |
| **Random Forest**       | **62,59%**            | **0.6834**  | **0.6226**   | 65,22%                          | 60,78%                    |
| **XGBoost (Campeao)**   | 62,51%                | 0.6833      | 0.6219       | **65,25%**                      | 60,61%                    |

## 5. Decisao de Negocio e Threshold Tuning (0,55)

Em politicas publicas de Busca Ativa, o custo de um Falso Negativo (deixar de identificar uma crianca que necessita de apoio pedagogico) e significativamente maior do que o custo de um Falso Positivo.

Com o ajuste do limiar operacional no modelo XGBoost para 0,55, obteve-se o seguinte ganho:

| **Limiar de Corte (Threshold)** | **Recall (Alunos Nao Alfabetizados / Risco)** | **Impacto em Politicas Publicas**                                  |
| ------------------------------- | --------------------------------------------- | ------------------------------------------------------------------ |
| **0,50 (Padrao Tecnico)**       | 65,25%                                        | Captura 65 em cada 100 alunos em risco.                            |
| **0,55 (Regra de Negocio)**     | **77,10%**                                    | **Captura 77 em cada 100 alunos em risco (+11,85% de cobertura).** |

## 6. Limitacoes Reais e Eticas

1. **Agregacao de Dados Territoriais:** Em conformidade com a LGPD e o protocolo de sigilo do INEP, dados brutos foram agregados por municipio, impondo um limite natural de acuracia global (~62.5%).
2. **Uso Exclusivo para Suporte:** O modelo deve ser utilizado estritamente para alerta precoce e triagem distributiva de recursos de apoio, nunca para fins punitivos ou discriminatorios.

## 7. Como Executar o Projeto

Bash

```bash
# 1. Ativar o ambiente virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Gerar Analise Exploratoria (Salva graficos em reports/images/)
python -m src.ml.exploratory_analysis

# 3. Executar o Treinamento Completo (Salva artefatos em models/)
python -m src.ml.train

# 4. Gerar Graficos de Explicabilidade (SHAP / Gain)
python -m src.ml.explain

# 5. Executar o Modulo de Inferencia
python -m src.ml.predict
```

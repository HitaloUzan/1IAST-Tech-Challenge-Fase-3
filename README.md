# Tech Challenge - Fase 3: Predicao da Alfabetizacao Infantil e Busca Ativa

## 1. Contexto e Objetivos de Negocio

A alfabetizacao na idade certa e um dos principais desafios da educacao basica publica no Brasil. Este projeto desenvolve uma pipeline de Machine Learning integrada a arquitetura GCP (BigQuery - Camada Gold, construida na Fase 2) para prever a probabilidade de um aluno estar alfabetizado, servindo como apoio a decisao para politicas de Busca Ativa e Reforco Escolar.

## 2. Origem e colaboracao

Este repositorio e um fork de [NaiaraMartins/1IAST-Tech-Challenge-Fase-3](https://github.com/NaiaraMartins/1IAST-Tech-Challenge-Fase-3), que construiu a base de dados (gold.ml_features_alunos, integrando silver.alunos_clean + silver.inse_escola_clean), a EDA, o pipeline sklearn inicial e a explicabilidade (SHAP). Este fork adiciona:

- Otimizacao de hiperparametros com Optuna (ausente na v1 -- os 3 modelos usavam parametros fixos).
- Split e cross-validation agrupados por id_aluno em vez de linha aleatoria -- ver secao 4.
- Pipeline sklearn unico (preprocessor + undersampler + modelo em um so objeto), em vez de preprocessor e modelo salvos separados.
- Uma tabela Gold propria, gold.ml_features_alunos_v2 (mesma logica de join da v1, com id_aluno preservado), para nao sobrescrever a tabela original no mesmo projeto GCP compartilhado.

## 3. Estrutura do Projeto

```text
config.py                   # Credenciais GCP e constantes globais
requirements.txt
reports/
  images/                   # EDA, Feature Importance, SHAP
  model_results.csv         # Comparativo de metricas (teste agrupado por aluno)
  optuna_best_params.json   # Hiperparametros vencedores por modelo
  threshold_simulation_*.csv
models/                     # Pipelines treinados (.joblib) -- nao versionado
src/
  gold/build_gold_ml.py     # Cria gold.ml_features_alunos_v2 no BigQuery
  ml/
    exploratory_analysis.py
    preprocessing.py        # Split agrupado, Pipeline unico
    tune.py                 # Busca de hiperparametros com Optuna
    train.py                # Treino final + comparativo
    evaluate.py              # Simulacao de threshold (regra de negocio)
    predict.py               # Inferencia
    explain.py                # SHAP / Feature Importance
```

## 4. Protecao contra Data Leakage

- Filtro de escopo: apenas redes publicas (Municipal e Estadual), como na v1.
- proficiencia (nota SAEB) excluida das features: e a mesma variavel que define o rotulo alfabetizado (corte oficial de 743 pontos), confirmado via query direta no BigQuery -- alfabetizado = 'Sim' equivale a proficiencia >= 743 de forma exata nos dados.
- Split e CV agrupados por id_aluno (GroupShuffleSplit / StratifiedGroupKFold): descobrimos, consultando gold.ml_features_alunos_v2, que 51,2% dos alunos aparecem em 2 linhas (edicoes 2023 e 2024). Um train_test_split aleatorio por linha -- usado na v1 -- deixa o mesmo aluno em treino e teste ao mesmo tempo. Corrigido aqui: nenhum aluno do treino aparece no teste (verificado programaticamente a cada execucao de prepare_data()).
- Isolamento dos transformadores: SimpleImputer/StandardScaler/OneHotEncoder e RandomUnderSampler vivem dentro do mesmo Pipeline do modelo, com fit restrito a cada fold/treino -- nunca veem o conjunto de teste.

## 5. Otimizacao de Hiperparametros (Optuna)

Cada modelo (Regressao Logistica, Random Forest, XGBoost) tem um study Optuna proprio: TPESampler + MedianPruner, StratifiedGroupKFold (agrupado por aluno) dentro do objective, storage persistente em SQLite (reports/optuna_study.db, recriavel via python -m src.ml.tune).

A busca roda em uma subamostra de 150.000 linhas (em vez dos ~2,68M de treino): cada trial treina N modelos, um por fold, e em milhoes de linhas isso inviabilizaria dezenas de trials em tempo de desenvolvimento local. O split final e o treino do modelo campeao usam a base completa. Decisao de tempo/FinOps documentada, no mesmo espirito das decisoes de particionamento da Fase 2.

| Modelo | ROC-AUC (CV, subamostra) | Melhores parametros |
|---|---|---|
| Logistic | 0.6701 | C=13.20 |
| Random Forest | 0.6732 | n_estimators=299, max_depth=9, min_samples_leaf=6, max_features=sqrt |
| XGBoost | 0.6737 | n_estimators=489, max_depth=6, learning_rate=0.0101, subsample=0.79, colsample_bytree=0.77, min_child_weight=4, reg_lambda=2.55 |

## 6. Resultados Finais (teste agrupado por aluno, ~670.817 registros)

| Modelo | Accuracy | F1-Macro | Recall (Nao Alfab.) | Recall (Alfabetizado) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic | 61,81% | 0.6138 | 62,75% | 61,16% | 0.6712 |
| Random Forest (campeao) | 62,17% | 0.6181 | 64,21% | 60,76% | 0.6770 |
| XGBoost | 62,13% | 0.6177 | 64,10% | 60,77% | 0.6767 |

O Random Forest venceu por margem minima sobre o XGBoost (+0,0003 ROC-AUC) -- estatisticamente, os tres modelos convergem para o mesmo teto de sinal disponivel nos dados (ver secao 9).

### Comparacao metodologica com a v1 (Naiara)

| Modelo | ROC-AUC v1 (split por linha) | ROC-AUC aqui (split por aluno) | Diferenca |
|---|---|---|---|
| Logistic | 0.6723 | 0.6712 | -0.0011 |
| Random Forest | 0.6834 | 0.6770 | -0.0064 |
| XGBoost | 0.6833 | 0.6767 | -0.0066 |

A v1 tinha ROC-AUC mais alto -- mas isso e justamente o efeito esperado do vazamento descrito na secao 4: o split aleatorio por linha deixa o mesmo aluno (repetido em 2023/2024) influenciar treino e teste ao mesmo tempo, inflando a metrica de forma mais visivel nos modelos de arvore (mais flexiveis para explorar a quase-duplicata) do que na regressao logistica (mais regularizada). O numero mais baixo aqui e o mais confiavel para generalizacao real.

## 7. Interpretabilidade (Random Forest, modelo campeao)

### Feature Importance (Gini)

| Variavel | Contribuicao (%) |
|---|---|
| taxa_alfabetizacao_municipio | 38,08% |
| media_portugues_municipio | 29,38% |
| proporcao_adequado_avancado | 14,15% |
| proporcao_basico | 8,21% |
| proporcao_abaixo_basico | 4,13% |
| peso_aluno | 2,97% |
| inse_municipio | 1,55% |
| rede_Estadual | 0,82% |
| rede_Municipal | 0,71% |

O historico municipal de alfabetizacao (taxa_alfabetizacao_municipio + media_portugues_municipio, juntas ~67%) domina a predicao -- esperado, ja que sao as variaveis mais diretamente correlacionadas ao desfecho no nivel agregado disponivel.

### SHAP

reports/images/shap_summary_random_forest.png (impacto direcional) e shap_waterfall_random_forest.png (explicacao individual) confirmam a mesma hierarquia: taxa e media municipais deslocam a probabilidade de alfabetizacao positivamente; INSE tem efeito na mesma direcao, porem secundario.

## 8. Aplicacao Estrategica: Threshold Tuning para Busca Ativa

Em politicas de Busca Ativa, o custo de um Falso Negativo (nao identificar uma crianca que precisa de apoio) e maior que o de um Falso Positivo. Simulacao de threshold no modelo campeao:

| Threshold | Accuracy | Recall (risco) | Precision (risco) |
|---|---|---|---|
| 0,50 (padrao) | 62,17% | 64,21% | 53,06% |
| 0,55 | 59,38% | 75,68% | 50,19% |
| 0,60 | 55,29% | 86,32% | 47,41% |
| 0,65 | 51,75% | 92,52% | 45,54% |

Com threshold 0,55 (mesma regra de negocio adotada na v1), o recall de alunos em risco sobe de 64,2% para 75,7% -- um ganho de ~11,5pp de cobertura, ao custo de mais falsos positivos (triagem mais ampla, adequada quando o custo de reforco extra e menor que o de nao intervir).

## 9. Limitacoes Reais e Eticas

1. Grao dos dados: a base publica do INEP (Base dos Dados) so libera microdado individual (id_aluno) sem variaveis socioeconomicas no mesmo grao -- o contexto territorial/socioeconomico (inse_municipio, taxas municipais) e agregado por municipio. Isso impoe um teto de sinal: ROC-AUC ~0,67-0,68 e accuracy ~62% sao consistentes nos tres modelos e nas duas versoes do pipeline, sugerindo que o limite e do dado disponivel, nao do modelo ou do tuning.
2. Correlacao ecologica: usar taxas municipais como preditor de um desfecho individual carrega o risco classico de inferencia ecologica -- o modelo aprende "o municipio deste aluno historicamente vai bem/mal", nao necessariamente fatores causais sobre o aluno especifico.
3. Uso exclusivo para suporte: o modelo deve ser usado para alerta precoce e triagem distributiva de recursos, nunca para fins punitivos ou discriminatorios.

## 10. Aplicacao Pratica para Politicas Publicas

O modelo, com o threshold ajustado (0,55-0,60), pode alimentar paineis de Busca Ativa priorizando municipios/escolas de rede publica com maior proporcao de alunos em risco previsto, direcionando reforco escolar e recursos do FUNDEB de forma mais eficiente que uma triagem uniforme.

## 11. Possiveis Evolucoes Futuras

- Enriquecer com fontes externas no grao municipio (Censo Escolar, PNAD, Atlas do Desenvolvimento Humano) para reduzir a limitacao da secao 9.2.
- Rodar a busca Optuna na base de treino completa (nao a subamostra) em ambiente com mais capacidade computacional, e aumentar OPTUNA_N_TRIALS.
- Testar StratifiedGroupKFold por id_escola (alem de id_aluno) para avaliar generalizacao entre escolas nao vistas.
- Modelo de serie temporal (2023 -> 2024) para prever evolucao ano a ano por municipio.

## 12. Como Executar

```bash
python -m venv .venv
.venv\Scripts\activate  # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Credenciais: coloque o service-account.json (mesmo projeto GCP da Fase 2) em credentials/
# export GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json

python -m src.gold.build_gold_ml          # cria gold.ml_features_alunos_v2
python -m src.ml.exploratory_analysis     # EDA -> reports/images/
python -m src.ml.tune                     # busca Optuna (ou --model <nome> por vez)
python -m src.ml.train                    # treino final + reports/model_results.csv
python -m src.ml.evaluate --model random_forest.joblib   # threshold tuning
python -m src.ml.explain --model random_forest.joblib    # SHAP / feature importance
python -m src.ml.predict                  # inferencia de exemplo
```
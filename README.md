# Tech Challenge - Fase 3: Predicao da Alfabetizacao Infantil e Busca Ativa

## 1. Contexto e Objetivos de Negocio

A alfabetizacao na idade certa e um dos principais desafios da educacao basica publica no Brasil. Este projeto desenvolve uma pipeline de Machine Learning integrada a arquitetura GCP (BigQuery - Camada Gold, construida na Fase 2) para prever a probabilidade de um aluno estar alfabetizado, servindo como apoio a decisao para politicas de Busca Ativa e Reforco Escolar.

## 2. Origem e colaboracao

Este repositorio e um fork de [NaiaraMartins/1IAST-Tech-Challenge-Fase-3](https://github.com/NaiaraMartins/1IAST-Tech-Challenge-Fase-3), que construiu a base de dados (gold.ml_features_alunos, integrando silver.alunos_clean + silver.inse_escola_clean), a EDA, o pipeline sklearn inicial e a explicabilidade (SHAP). Este fork adiciona:

- Otimizacao de hiperparametros com Optuna (ausente na v1 -- os 3 modelos usavam parametros fixos).
- Split e cross-validation agrupados por id_aluno em vez de linha aleatoria -- ver secao 4.
- Pipeline sklearn unico (preprocessor + modelo em um so objeto), em vez de preprocessor e modelo salvos separados.
- Correcao de um vazamento temporal: a v1 calculava a taxa de alfabetizacao do municipio misturando 2023+2024 (media unica) para toda linha, entao uma linha de 2023 recebia informacao de 2024, que ainda nao existia. Corrigido para casar pelo mesmo ano da linha.
- Feature nova no grao de escola (mais fino que municipio): taxa de alfabetizacao da propria escola no ano anterior, calculada a partir de alunos_clean, sem risco de vazamento (nunca usa o mesmo ano/mesma turma).
- Feature nova: codigo de UF (2 primeiros digitos do id_municipio).
- Troca de RandomUnderSampler por class_weight/scale_pos_weight -- a v1 descartava boa parte da classe majoritaria a cada fold; aqui o balanceamento e feito sem jogar dado fora.
- Uma tabela Gold propria, gold.ml_features_alunos_v2, para nao sobrescrever a tabela original no mesmo projeto GCP compartilhado.

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
- Split e CV agrupados por id_aluno (GroupShuffleSplit / StratifiedGroupKFold): descobrimos, consultando a gold, que 51,2% dos alunos aparecem em 2 linhas (edicoes 2023 e 2024). Um train_test_split aleatorio por linha -- usado na v1 -- deixa o mesmo aluno em treino e teste ao mesmo tempo. Corrigido aqui: nenhum aluno do treino aparece no teste (verificado programaticamente a cada execucao de prepare_data()).
- Taxa/media/proporcoes do municipio casadas pelo MESMO ano da linha (nao mais uma media borrada entre 2023 e 2024 -- ver secao 2).
- Taxa de alfabetizacao da escola usa APENAS o ano anterior (ano-1): nunca inclui o proprio aluno nem sua turma. Cobertura: ~79% das linhas de 2024 tem historico de 2023; linhas de 2023 ficam sem essa feature (nao ha 2022 nos dados) -- sinalizado pela flag tem_historico_escola em vez de fabricar um valor.
- Isolamento dos transformadores: SimpleImputer/StandardScaler/OneHotEncoder vivem dentro do mesmo Pipeline do modelo, com fit restrito a cada fold/treino -- nunca veem o conjunto de teste.

## 5. Otimizacao de Hiperparametros (Optuna)

Cada modelo (Regressao Logistica, Random Forest, XGBoost) tem um study Optuna proprio: TPESampler + MedianPruner, StratifiedGroupKFold (agrupado por aluno) dentro do objective, storage persistente em SQLite (reports/optuna_study.db, recriavel via python -m src.ml.tune).

A busca roda em uma subamostra de 150.000 linhas (em vez dos ~2,68M de treino): cada trial treina N modelos, um por fold, e em milhoes de linhas isso inviabilizaria dezenas de trials em tempo de desenvolvimento local. O split final e o treino do modelo campeao usam a base completa. Decisao de tempo/FinOps documentada, no mesmo espirito das decisoes de particionamento da Fase 2.

| Modelo | ROC-AUC (CV, subamostra) | Melhores parametros |
|---|---|---|
| Logistic | 0.6819 | C=0.0060 |
| Random Forest | 0.6840 | n_estimators=176, max_depth=11, min_samples_leaf=4, max_features=sqrt |
| XGBoost | 0.6843 | n_estimators=489, max_depth=6, learning_rate=0.0101, subsample=0.79, colsample_bytree=0.77, min_child_weight=4, reg_lambda=2.55 |

## 6. Resultados Finais (teste agrupado por aluno, ~670.817 registros)

| Modelo | Accuracy | F1-Macro | Recall (Nao Alfab.) | Recall (Alfabetizado) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic | 62,63% | 0.6220 | 63,65% | 61,92% | 0.6820 |
| Random Forest (campeao) | 62,68% | 0.6235 | 65,15% | 60,98% | 0.6866 |
| XGBoost | 62,67% | 0.6235 | 65,41% | 60,77% | 0.6860 |

Random Forest venceu por margem minima sobre o XGBoost (+0,0006 ROC-AUC) -- de novo, os tres modelos convergem para o mesmo teto de sinal disponivel (ver secao 9).

### Evolucao do pipeline (3 versoes, mesmo teste agrupado por aluno)

| Modelo | v1 Naiara (split por linha, sem tuning) | Fork -- so leakage fix (Optuna + split por aluno) | Fork -- + features de escola/UF/ano | Ganho total |
|---|---|---|---|---|
| Logistic | 0.6723 | 0.6712 | 0.6820 | +0.0097 |
| Random Forest | 0.6834 | 0.6770 | 0.6866 | +0.0032 |
| XGBoost | 0.6833 | 0.6767 | 0.6860 | +0.0027 |

A primeira correcao (split agrupado) reduziu o ROC-AUC porque removeu o vazamento entre treino/teste da v1 -- numero mais baixo, porem confiavel. A segunda rodada (features de escola/UF + fix do blend de anos) recuperou e superou o numero original da v1, desta vez sem vazamento. Accuracy tambem subiu de ~62,2% (fork inicial) para ~62,7%.

## 7. Interpretabilidade (Random Forest, modelo campeao)

### Feature Importance (Gini)

| Variavel | Contribuicao (%) |
|---|---|
| taxa_alfabetizacao_municipio | 32,93% |
| media_portugues_municipio | 28,12% |
| proporcao_basico | 8,46% |
| proporcao_adequado_avancado | 7,72% |
| proporcao_abaixo_basico | 4,99% |
| UF (Santa Catarina) | 4,25% |
| inse_municipio | 2,86% |
| peso_aluno | 2,52% |
| UF (Bahia) | 2,47% |
| taxa_alfabetizacao_escola_prior | 0,77% |
| rede / n_alunos_prior_escola / tem_historico_escola / demais UFs | restante |

O historico municipal de alfabetizacao ainda domina (~61% somado), mas agora responde por uma fatia menor do total (era ~67% na v2 sem as novas features) -- UF e as proporcoes por nivel ganharam peso relativo. Curiosamente, a taxa historica da propria escola teve contribuicao individual pequena (0,77%) apesar de ter ajudado o ROC-AUC geral -- sinal de que o efeito dela e mais forte em interacao com outras variaveis do que isoladamente (algo que uma arvore de profundidade limitada nao captura bem sozinha).

### SHAP

reports/images/shap_summary_random_forest.png (impacto direcional) e shap_waterfall_random_forest.png (explicacao individual) confirmam a mesma hierarquia: taxa e media municipais deslocam a probabilidade de alfabetizacao positivamente; INSE e UF tem efeito secundario, porem visivel.

## 8. Aplicacao Estrategica: Threshold Tuning para Busca Ativa

Em politicas de Busca Ativa, o custo de um Falso Negativo (nao identificar uma crianca que precisa de apoio) e maior que o de um Falso Positivo. Simulacao de threshold no modelo campeao:

| Threshold | Accuracy | Recall (risco) | Precision (risco) |
|---|---|---|---|
| 0,50 (padrao) | 62,68% | 65,15% | 53,56% |
| 0,55 | 59,94% | 77,02% | 50,64% |
| 0,60 | 55,81% | 87,31% | 47,76% |
| 0,65 | 52,68% | 92,37% | 46,05% |

Com threshold 0,55 (mesma regra de negocio adotada na v1), o recall de alunos em risco sobe de 65,2% para 77,0% -- um ganho de ~11,9pp de cobertura, ao custo de mais falsos positivos.

## 9. Limitacoes Reais e Eticas

1. Grao dos dados: mesmo apos adicionar historico de escola e UF, a maior parte do sinal ainda vem de agregados territoriais (municipio) -- tentamos ligar INSE no grao de escola diretamente, mas o id_escola de silver.inse_escola_clean nao compartilha o mesmo namespace de codigo do id_escola em silver.alunos_clean (0 de 42.802 escolas batem apos normalizar), entao esse enriquecimento ficou fora de alcance com os dados atuais. O teto de sinal (ROC-AUC ~0,68-0,69, accuracy ~62-63%) e consistente nas 3 versoes do pipeline, sugerindo que o limite e estrutural, nao de modelo ou tuning.
2. Correlacao ecologica: usar taxas municipais como preditor de um desfecho individual carrega o risco classico de inferencia ecologica.
3. Uso exclusivo para suporte: o modelo deve ser usado para alerta precoce e triagem distributiva de recursos, nunca para fins punitivos ou discriminatorios.

## 10. Aplicacao Pratica para Politicas Publicas

O modelo, com o threshold ajustado (0,55-0,60), pode alimentar paineis de Busca Ativa priorizando municipios/escolas de rede publica com maior proporcao de alunos em risco previsto, direcionando reforco escolar e recursos do FUNDEB de forma mais eficiente que uma triagem uniforme.

## 11. Possiveis Evolucoes Futuras

- Resolver a divergencia de codigo de escola entre silver.alunos_clean e silver.inse_escola_clean (provavelmente exige uma tabela de-para externa, ex. Censo Escolar/INEP) para habilitar INSE no grao de escola.
- Enriquecer com fontes externas no grao municipio (Censo Escolar, PNAD, Atlas do Desenvolvimento Humano).
- Rodar a busca Optuna na base de treino completa (nao a subamostra) em ambiente com mais capacidade computacional, e aumentar OPTUNA_N_TRIALS.
- Testar StratifiedGroupKFold por id_escola (alem de id_aluno) para avaliar generalizacao entre escolas nao vistas.
- Modelo de serie temporal (2023 -> 2024) para prever evolucao ano a ano por municipio, agora que ja existe a infraestrutura de features "ano-1".

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
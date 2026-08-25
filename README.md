# Tech Challenge - Fase 3: Predicao da Alfabetizacao Infantil e Busca Ativa

**PosTech FIAP | IA Science**

## 1. Contexto do Problema

A alfabetizacao na idade certa e um dos principais desafios da educacao basica publica no Brasil. O **Compromisso Nacional Crianca Alfabetizada** estabelece que todas as criancas estejam alfabetizadas ao final do 2o ano do Ensino Fundamental ate 2030, medido pelo **Indicador Crianca Alfabetizada** do INEP (corte de 743 pontos na escala SAEB).

Compreender apenas os dados atuais nao basta: gestores publicos precisam **antecipar riscos**, identificar regioes vulneraveis e entender quais fatores mais impactam o indicador.

## 2. Objetivo Analitico

Desenvolver um modelo supervisionado capaz de prever se um aluno sera considerado **alfabetizado ou nao alfabetizado**, usando variaveis educacionais, territoriais e socioeconomicas provenientes da camada **Gold** construida na Fase 2, e transformar essa predicao em inteligencia aplicada a politicas de **Busca Ativa** e **Reforco Escolar**.

## 3. Origem e Colaboracao

Fork de [NaiaraMartins/1IAST-Tech-Challenge-Fase-3](https://github.com/NaiaraMartins/1IAST-Tech-Challenge-Fase-3), que construiu a base de dados, a EDA e o pipeline sklearn inicial. Este fork adiciona otimizacao com Optuna, corrige tres fontes de vazamento, incorpora as metas oficiais e responde as cinco perguntas de negocio do desafio (secao 9).

## 4. Descricao da Base Utilizada

Tabela `gold.ml_features_alunos_v2`, com grao de **aluno** (`id_aluno`), 3.354.661 registros das edicoes 2023 e 2024, construida por `src/preprocessing/build_gold_ml.py` a partir das camadas silver/gold da Fase 2:

| Fonte (Fase 2) | Contribuicao |
|---|---|
| `silver.alunos_clean` | Grao de aluno, rotulo `alfabetizado`, rede, peso amostral |
| `silver.alfabetizacao_municipio_clean` | Indicador Crianca Alfabetizada, media de portugues, proporcoes por nivel |
| `silver.inse_escola_clean` | Indicador socioeconomico (INSE), agregado por municipio |
| `silver.metas_consolidadas` | Metas municipais 2024-2030, participacao, nivel de alfabetizacao |
| `id_municipio` (IBGE) | Dados territoriais: municipio e UF |

### Features do modelo

| Grupo | Variaveis |
|---|---|
| Educacionais | `taxa_alfabetizacao_municipio`, `media_portugues_municipio`, `proporcao_abaixo_basico`, `proporcao_basico`, `proporcao_adequado_avancado` |
| Socioeconomicas | `inse_municipio` |
| Metas | `meta_2024`, `percentual_participacao`, `nivel_alfabetizacao` |
| Temporais | `taxa_alfabetizacao_escola_prior`, `n_alunos_prior_escola`, `tem_historico_escola` |
| Territoriais | `sigla_uf_code`, `rede` |
| Amostral | `peso_aluno` |

## 5. Estrutura do Projeto

```text
data/                        # Dados locais (nao versionados)
notebooks/
  01_analise_exploratoria.ipynb
images/                      # Graficos: EDA, Feature Importance, SHAP, perguntas de negocio
reports/                     # CSVs de metricas e analises
src/
  preprocessing/
    build_gold_ml.py         # Constroi gold.ml_features_alunos_v2 no BigQuery
    features.py              # Split agrupado, ColumnTransformer, Pipeline unico
  modeling/
    tune.py                  # Otimizacao de hiperparametros (Optuna)
    train.py                 # Treino final e comparativo
    predict.py               # Motor de inferencia
  evaluation/
    evaluate.py              # Threshold tuning (regra de negocio)
    explain.py               # Feature Importance + SHAP
    business_questions.py    # Perguntas de negocio (secao 9)
  visualization/
    eda_plots.py             # Graficos da analise exploratoria
config.py
requirements.txt
```

## 6. Etapas de Modelagem

1. **Camada Gold ML** (`build_gold_ml.py`): integra as fontes acima no grao de aluno.
2. **Analise exploratoria** (`notebooks/01_analise_exploratoria.ipynb` + `eda_plots.py`): distribuicoes, correlacoes, nulos e formulacao das hipoteses H1-H4.
3. **Pipeline de pre-processamento** integrado ao modelo em um unico objeto sklearn:
   - `SimpleImputer(median)` + `StandardScaler` nas numericas;
   - `SimpleImputer(most_frequent)` + `OneHotEncoder` nas categoricas;
   - balanceamento via `class_weight` / `scale_pos_weight`.
4. **Otimizacao** (`tune.py`): Optuna com `StratifiedGroupKFold`.
5. **Treino e avaliacao** (`train.py`, `evaluate.py`) na base completa.
6. **Interpretabilidade** (`explain.py`): Feature Importance + SHAP.
7. **Aplicacao estrategica** (`business_questions.py`).

## 7. Tratamento de Data Leakage

Esta foi a area de maior esforco do projeto. Cinco fontes de vazamento foram identificadas e tratadas:

### 7.1 `proficiencia` define o rotulo
`alfabetizado = 'Sim'` equivale exatamente a `proficiencia >= 743` (corte SAEB), confirmado por query direta. A variavel foi **excluida das features**.

### 7.2 Mesmo aluno em treino e teste
Consulta a base revelou que **51,2% dos alunos aparecem em duas linhas** (edicoes 2023 e 2024). Um `train_test_split` aleatorio por linha colocaria o mesmo aluno nos dois conjuntos. Todo o projeto usa `GroupShuffleSplit` / `StratifiedGroupKFold` **agrupados por `id_aluno`**, com verificacao automatica de sobreposicao a cada execucao (`Alunos em comum: 0`).

### 7.3 Mistura de anos nos agregados municipais
A versao original calculava a media de `taxa_alfabetizacao` misturando 2023 e 2024 para todas as linhas -- uma linha de 2023 recebia informacao de 2024. Corrigido para casar pelo **mesmo ano** da linha.

### 7.4 Historico de escola do mesmo ano
`taxa_alfabetizacao_escola_prior` usa exclusivamente o **ano anterior** (`ano - 1`), nunca a mesma turma/edicao. Cobertura: ~79% das linhas de 2024; linhas de 2023 ficam sem historico (nao ha 2022 na base) e isso e sinalizado pela flag `tem_historico_escola`, em vez de fabricar valor.

### 7.5 Vazamento same-cohort via TargetEncoder (testado e revertido)

Aplicamos `sklearn.preprocessing.TargetEncoder` em `id_municipio` e `id_escola` esperando ganho legitimo. O ROC-AUC de teste subiu de 0,6820 para 0,7032 -- mas **acima do ROC-AUC da validacao cruzada (0,6840)**, sinal classico de vazamento. Teste controlado no mesmo split:

| Configuracao | ROC-AUC teste |
|---|---|
| Sem target encoding | 0,6820 |
| Apenas `id_municipio` | 0,6821 (+0,0001, ruido) |
| `id_municipio` + `id_escola` | 0,7032 (+0,0212) |

**Todo o ganho vinha de `id_escola`.** O cross-fitting interno do TargetEncoder protege contra vazar o target da propria linha, mas nao contra vazar o target dos **colegas de escola medidos na mesma edicao do SAEB** -- informacao indisponivel numa predicao real, onde a prova daquele ano ainda nao ocorreu. A tecnica foi **revertida**, e o efeito legitimo equivalente permanece capturado por `taxa_alfabetizacao_escola_prior` (ano anterior).

### 7.6 Features redundantes removidas

`meta_2030` tem **um unico valor distinto** na base (80,0 para todos os municipios) e recebeu 0,0% de importancia. `gap_meta_2030` = `taxa_alfabetizacao_municipio - 80,0`, portanto **correlacao 1,0000** com ela. Mantidas, apareciam com 17,75% de "importancia" que na verdade era a taxa municipal reapresentada. Foram removidas das features (permanecem na gold, alimentando as analises de negocio). O ROC-AUC ficou inalterado (0,6882 -> 0,6881), confirmando que nao carregavam informacao.

## 8. Escolha do Algoritmo, Otimizacao e Metricas

### Otimizacao (Optuna)

`TPESampler` + `MedianPruner`, com `StratifiedGroupKFold` agrupado por aluno dentro do objective e storage persistente em SQLite. A busca roda em subamostra de 150.000 linhas (cada trial treina um modelo por fold; em 2,68M linhas isso inviabilizaria dezenas de trials localmente) -- o **treino final usa a base completa**.

| Modelo | ROC-AUC (CV) | Melhores parametros |
|---|---|---|
| Logistic | 0,6830 | `C=0.00195` |
| Random Forest | 0,6853 | `n_estimators=376, max_depth=11, min_samples_leaf=4, max_features=sqrt` |
| XGBoost | 0,6860 | `n_estimators=489, max_depth=6, learning_rate=0.0101, subsample=0.79, colsample_bytree=0.77, min_child_weight=4, reg_lambda=2.55` |

### Resultados finais (teste agrupado por aluno, 670.817 registros)

| Modelo | Accuracy | F1-Macro | Recall (Nao Alfab.) | Recall (Alfabetizado) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic | 62,40% | 0,6208 | 65,28% | 60,41% | 0,6828 |
| **Random Forest (campeao)** | 62,44% | 0,6222 | **67,08%** | 59,24% | **0,6881** |
| XGBoost | 62,67% | 0,6239 | 66,16% | 60,25% | 0,6879 |

**Validacao da ausencia de vazamento:** o ROC-AUC de teste (0,6881) esta praticamente colado ao da validacao cruzada (0,6853). Na versao com TargetEncoder, o teste ficava 2 pontos **acima** da CV -- a assinatura do vazamento que foi corrigido.

O **Random Forest** foi escolhido como campeao por combinar o maior ROC-AUC com o maior recall da classe de risco (67,08%), que e a metrica operacionalmente relevante para Busca Ativa.

## 9. Aplicacao Estrategica: as cinco perguntas de negocio

Executavel por `python -m src.evaluation.business_questions`.

### 9.1 Quais fatores mais impactam a alfabetizacao? / 9.5 Quais variaveis tem maior influencia nos modelos?

| Variavel | Contribuicao |
|---|---|
| `taxa_alfabetizacao_municipio` | 23,76% |
| `nivel_alfabetizacao` | 19,89% |
| `media_portugues_municipio` | 16,29% |
| `meta_2024` | 12,87% |
| `proporcao_adequado_avancado` | 6,02% |
| `proporcao_basico` | 4,67% |
| `proporcao_abaixo_basico` | 3,50% |
| UF (Ceara) | 2,16% |
| `peso_aluno` | 1,96% |
| `inse_municipio` | 1,82% |

O contexto educacional do municipio domina (~53% somando taxa, media de portugues e nivel). As **metas oficiais** contribuem com ~14% (`nivel_alfabetizacao` + `meta_2024` + participacao), confirmando que a trajetoria pactuada carrega sinal proprio. O INSE tem efeito real, porem secundario -- o territorio importa mais que a renda isolada.

Os graficos SHAP (`images/shap_summary_random_forest.png` e `shap_waterfall_random_forest.png`) confirmam a direcao: maiores taxas municipais e maior nivel socioeconomico deslocam positivamente a probabilidade de alfabetizacao.

### 9.2 Quais municipios apresentam maior risco educacional?

Agregando o risco previsto (`1 - P(alfabetizado)`) por municipio, entre 4.177 municipios com pelo menos 20 alunos avaliados no conjunto de teste:

| UF | Municipios | Risco medio |
|---|---|---|
| SE | 61 | 0,737 |
| BA | 392 | 0,702 |
| RN | 107 | 0,679 |
| TO | 84 | 0,650 |
| AP | 16 | 0,624 |
| PA | 143 | 0,608 |

Ranking completo em `reports/q2_ranking_municipios_risco.csv`; grafico em `images/q2_municipios_risco.png`. A concentracao no Norte/Nordeste e consistente com a geografia educacional conhecida do pais.

### 9.3 Quais regioes possuem padroes semelhantes?

KMeans (k=4) sobre risco previsto, taxa real, INSE, gap ate a meta e participacao:

| Perfil | Municipios | Risco medio | Taxa alfabetizacao | INSE | Gap ate meta |
|---|---|---|---|---|---|
| **Critico** | 855 | 0,701 | 36,7% | 4,42 | -42,7 pp |
| **Atencao** | 791 | 0,520 | 56,2% | 4,37 | -23,8 pp |
| **Intermediario** | 1.437 | 0,441 | 64,2% | 5,21 | -15,9 pp |
| **Consolidado** | 922 | 0,228 | 83,7% | 4,76 | +2,9 pp |

**Insight nao obvio:** o INSE **nao** separa os grupos de forma monotonica -- o cluster Critico (4,42) tem INSE ligeiramente **maior** que o cluster Atencao (4,37), e o Consolidado (4,76) tem INSE menor que o Intermediario (5,21). Ou seja, a diferenca entre municipios criticos e consolidados **nao e explicada primariamente por renda**, e sim por fatores territoriais e de gestao educacional. Isso e relevante para politica publica: transferencia de renda isolada nao fecharia a lacuna.

Detalhamento em `reports/q3_perfil_clusters.csv` e `q3_municipios_por_cluster.csv`; grafico em `images/q3_clusters_regionais.png`.

### 9.4 Como prever municipios que podem nao atingir metas futuras?

Projecao ate 2030 comparando a taxa atual com a meta oficial e o ritmo anual necessario:

| Classificacao | Municipios | % |
|---|---|---|
| Risco alto de nao atingir | **494** | 12,3% |
| Risco moderado | 1.509 | 37,6% |
| Provavel atingir | 1.349 | 33,6% |
| Meta ja atingida | 659 | 16,4% |

**494 municipios precisariam evoluir mais de 7 pontos percentuais ao ano** ate 2030 -- ritmo sem precedente historico na serie. Os casos mais extremos exigiriam mais de 12 pp/ano (ex.: municipios em RN, BA, SE e TO partindo de taxas abaixo de 10%).

Lista completa em `reports/q4_municipios_risco_meta.csv`; grafico em `images/q4_projecao_metas.png`.

## 10. Decisao de Negocio: Threshold Tuning para Busca Ativa

Em Busca Ativa, o custo de um **Falso Negativo** (nao identificar uma crianca que precisa de apoio) e maior que o de um Falso Positivo (incluir na triagem quem nao precisava).

| Threshold | Accuracy | Recall (risco) | Precision (risco) |
|---|---|---|---|
| 0,50 (padrao tecnico) | 62,44% | 67,08% | 53,20% |
| **0,55 (regra de negocio)** | 60,17% | **76,82%** | 50,83% |
| 0,60 | 56,12% | 86,96% | 47,96% |
| 0,65 | 53,25% | 91,75% | 46,35% |

Com threshold **0,55**, a cobertura de alunos em risco sobe de 67,1% para 76,8% (**+9,7 pp**), ao custo de ~2,3 pp de acuracia -- troca vantajosa quando o custo de reforco escolar extra e menor que o de deixar uma crianca sem apoio.

## 11. Interpretacao dos Resultados e Insights

1. **O territorio pesa mais que a renda.** INSE responde por apenas 1,82% da importancia, enquanto o conjunto de variaveis municipais/territoriais passa de 50%. A clusterizacao reforca: municipios criticos e consolidados tem INSE praticamente equivalente.
2. **As metas carregam sinal proprio** (~14% da importancia). Municipios com metas iniciais mais baixas (`meta_2024`) e menor `nivel_alfabetizacao` sao sistematicamente mais frageis -- a pactuacao ja refletia a fragilidade estrutural.
3. **12,3% dos municipios estao em rota de nao cumprir a meta 2030** no ritmo atual.
4. **O historico da propria escola contribui pouco isoladamente** (0,62%), mas melhorou o ROC-AUC geral -- efeito de interacao, nao de forca marginal.
5. **Metodologia importa mais que metrica.** As correcoes de vazamento reduziram o ROC-AUC aparente de 0,7032 para 0,6881; o numero menor e o unico que representa a capacidade real de generalizacao.

## 12. Limitacoes do Projeto

1. **Grao das variaveis explicativas.** O rotulo e individual, mas quase todo o contexto e agregado por municipio: dentro do mesmo municipio e rede, alunos recebem valores identicos. Isso impoe um teto estrutural -- ROC-AUC ~0,69 e accuracy ~62% se repetem nos tres algoritmos e em todas as versoes do pipeline, indicando limite de informacao, nao de modelo.
2. **INSE por escola indisponivel.** `silver.inse_escola_clean` e `silver.alunos_clean` usam **namespaces distintos de `id_escola`** (0 de 42.802 escolas casam apos normalizacao). Seria o enriquecimento de maior potencial e exige uma tabela de-para externa (Censo Escolar).
3. **Correlacao ecologica.** Usar taxas municipais para prever desfecho individual implica o risco classico de inferencia ecologica: o modelo aprende "o municipio deste aluno historicamente vai bem/mal", nao causalidade sobre o aluno.
4. **Serie temporal curta.** Apenas duas edicoes (2023, 2024). A projecao de metas e deliberadamente linear e auditavel -- qualquer modelo temporal mais sofisticado seria sobreajuste com dois pontos.
5. **Uso restrito a suporte.** O modelo deve orientar alerta precoce e alocacao de recursos, **nunca** decisoes punitivas, ranqueamento sancionatorio ou discriminacao de escolas, alunos ou gestores.

## 13. Aplicacao Pratica para Politicas Publicas

- **Priorizacao da Busca Ativa:** com threshold 0,55, o modelo identifica ~77% dos alunos em risco, permitindo direcionar visitas domiciliares e reforco escolar.
- **Alocacao de recursos (FUNDEB):** os 855 municipios do cluster **Critico** e os 494 em risco alto de nao atingir a meta 2030 formam uma lista objetiva de prioridade.
- **Alerta antecipado de metas:** a projecao permite agir antes de 2030, nao apenas constatar o descumprimento depois.
- **Politica alem da renda:** a evidencia de que o INSE nao separa os clusters sugere que intervencoes pedagogicas e de gestao tendem a ser mais efetivas que transferencia de renda isolada para fechar a lacuna de alfabetizacao.

## 14. Possiveis Evolucoes Futuras

- Obter tabela de-para de `id_escola` (Censo Escolar/INEP) para habilitar INSE e infraestrutura no grao de escola -- maior potencial de ganho identificado.
- Enriquecer com fontes externas municipais: PNAD, Atlas do Desenvolvimento Humano, Cadastro Unico, FUNDEB.
- Rodar a busca Optuna na base completa e ampliar `OPTUNA_N_TRIALS` em ambiente com mais capacidade.
- Validacao `StratifiedGroupKFold` por `id_escola` para medir generalizacao a escolas nunca vistas.
- Com tres ou mais edicoes, modelo de trajetoria temporal por municipio substituindo a projecao linear.

## 15. Como Executar

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Credenciais GCP (mesmo projeto da Fase 2) em credentials/service-account.json
export GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json

python -m src.preprocessing.build_gold_ml                  # camada Gold ML
python -m src.visualization.eda_plots                      # EDA -> images/
python -m src.modeling.tune                                # Optuna (ou --model <nome>)
python -m src.modeling.train                               # treino + reports/model_results.csv
python -m src.evaluation.evaluate --model random_forest.joblib
python -m src.evaluation.explain  --model random_forest.joblib
python -m src.evaluation.business_questions --model random_forest.joblib
python -m src.modeling.predict                             # inferencia de exemplo
```

O notebook `notebooks/01_analise_exploratoria.ipynb` documenta a analise exploratoria e as hipoteses que orientaram a modelagem.
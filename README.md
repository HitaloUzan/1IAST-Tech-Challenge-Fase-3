# Tech Challenge - Fase 3

Sistema de Engenharia de Dados e Machine Learning para análise da alfabetização infantil no Brasil utilizando Google Cloud Platform, BigQuery e Scikit-Learn.

## Objetivo

Este projeto tem como objetivo construir um pipeline completo de Engenharia de Dados e Machine Learning capaz de analisar dados de alfabetização do INEP, disponibilizando informações analíticas e criando um modelo preditivo para classificação de alunos alfabetizados.

## Tecnologias

- Python 3.13
- Google Cloud Platform
- BigQuery
- Google Cloud CLI
- Pandas
- Scikit-Learn
- SHAP
- Matplotlib
- Joblib

## Estrutura do Projeto
.
├── docs
├── images
├── notebooks
├── src
│   ├── gold
│   │   ├── build_gold.py
│   │   └── build_gold_ml.py
│   └── ml
│       ├── preprocessing.py
│       ├── train.py
│       ├── evaluate.py
│       ├── explain.py
│       └── predict.py
├── config.py
├── requirements.txt
└── README.md
## Arquitetura

O projeto segue uma arquitetura em camadas:

Bronze
↓

Silver
↓

Gold Analítica

Gold ML

Machine Learning

Predição
## Camada Bronze

Responsável pela ingestão dos dados originais.

## Camada Silver

Responsável pela limpeza, padronização e consolidação dos dados.

## Camada Gold

Responsável pela criação de indicadores analíticos utilizados para exploração dos dados.

## Camada Gold ML

Responsável pela construção do dataset de Machine Learning utilizado para treinamento do modelo.
## Tabelas Gold

- indicador_por_uf_ano
- ranking_estados
- perfil_desempenho_uf
- painel_municipios
- evolucao_temporal_brasil
- ml_features_alunos
## Dataset para Machine Learning

A tabela `gold.ml_features_alunos` foi criada especificamente para alimentar o pipeline de Machine Learning.

Ela reúne informações dos alunos juntamente com indicadores municipais provenientes das metas de alfabetização.

Principais atributos:

- ano
- id_municipio
- serie
- rede
- meta_alfabetizacao_2030
- percentual_participacao
- nivel_alfabetizacao
- possui_meta_municipal
- alfabetizado
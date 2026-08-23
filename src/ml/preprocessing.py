import logging
from typing import Tuple

import google.auth
from google.cloud import bigquery
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURAÇÃO OFICIAL DAS FEATURES DA CAMADA GOLD
# ============================================================

TARGET_COLUMN = getattr(config, "TARGET_COLUMN", "alfabetizado")

NUMERIC_FEATURES = [
    "taxa_alfabetizacao_municipio",
    "media_portugues_municipio",
    "proporcao_abaixo_basico",
    "proporcao_basico",
    "proporcao_adequado_avancado",
    "inse_municipio",
    "peso_aluno",
]

CATEGORICAL_FEATURES = [
    "rede",
]

# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def load_data(client: bigquery.Client) -> pd.DataFrame:
    """
    Carrega os dados corretos da camada Gold ML no BigQuery.
    """
    query = f"""
        SELECT
            taxa_alfabetizacao_municipio,
            media_portugues_municipio,
            proporcao_abaixo_basico,
            proporcao_basico,
            proporcao_adequado_avancado,
            inse_municipio,
            peso_aluno,
            rede,
            alfabetizado
        FROM `{config.GCP_PROJECT_ID}.{config.BQ_DATASET_GOLD}.{config.ML_FEATURES_TABLE}`
        WHERE rede IN ('Municipal', 'Estadual')
    """

    log.info("Carregando dados da Gold ML no BigQuery...")
    df = client.query(query).to_dataframe()
    log.info("Dados carregados com sucesso: %s registros", f"{len(df):,}")
    return df

# ============================================================
# SEPARAÇÃO BETWEEN FEATURES E TARGET
# ============================================================

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa as variáveis preditoras (X) do target (y) e realiza o mapeamento binário.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    
    # Mapeamento dinâmico para garantir 0 e 1
    y = df[TARGET_COLUMN].map({
        "Não": 0, "Sim": 1,
        "0": 0, "1": 1,
        0: 0, 1: 1
    })

    if y.isna().any():
        raise ValueError("A variável target possui valores nulos ou mapeamentos não reconhecidos.")

    return X, y

# ============================================================
# CONSTRUÇÃO DO PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Cria os pipelines de transformação (Imputer + Scaler / OneHot) para as colunas.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor

# ============================================================
# PREPARAÇÃO E DIVISION DOS DADOS
# ============================================================

def prepare_data(client: bigquery.Client, apply_undersampling: bool = True):
    """
    Executa a pipeline de extração, split treino/teste (80/20) e pré-processamento.
    Aplica RandomUnderSampler estritamente no treino para balancear em 50/50.
    """
    df = load_data(client)
    X, y = split_features_target(df)

    log.info(
        "Distribuição inicial do Target: 0=%s | 1=%s",
        f"{(y == 0).sum():,}",
        f"{(y == 1).sum():,}",
    )

    # Split Estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=getattr(config, "TEST_SIZE", 0.20),
        random_state=getattr(config, "RANDOM_STATE", 42),
        stratify=y,
    )

    log.info(
        "Divisão realizada -> Treino: %s | Teste: %s",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
    )

    preprocessor = build_preprocessor()

    # Fit SOMENTE no conjunto de treino (evita data leakage)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Balanceamento 50/50 restrito ao Treino
    if apply_undersampling:
        rus = RandomUnderSampler(random_state=getattr(config, "RANDOM_STATE", 42))
        X_train_processed, y_train = rus.fit_resample(X_train_processed, y_train)
        log.info("Balanceamento (50/50) via RandomUnderSampler aplicado com sucesso na base de treino.")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )

if __name__ == "__main__":
    credentials, _ = google.auth.default()
    client = bigquery.Client(
        project=config.GCP_PROJECT_ID,
        credentials=credentials,
    )
    prepare_data(client)
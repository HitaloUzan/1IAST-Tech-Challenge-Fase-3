import logging
from typing import Tuple

import pandas as pd
from google.cloud import bigquery
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TARGET_COLUMN = config.TARGET_COLUMN
GROUP_COLUMN = config.GROUP_COLUMN

NUMERIC_FEATURES = [
    "taxa_alfabetizacao_municipio",
    "media_portugues_municipio",
    "proporcao_abaixo_basico",
    "proporcao_basico",
    "proporcao_adequado_avancado",
    "inse_municipio",
    "peso_aluno",
]

CATEGORICAL_FEATURES = ["rede"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(client: bigquery.Client) -> pd.DataFrame:
    """
    Carrega gold.ml_features_alunos_v2 (mesma logica de join de
    NaiaraMartins/1IAST-Tech-Challenge-Fase-3, com id_aluno preservado).
    """
    query = f"""
        SELECT
            {GROUP_COLUMN},
            taxa_alfabetizacao_municipio,
            media_portugues_municipio,
            proporcao_abaixo_basico,
            proporcao_basico,
            proporcao_adequado_avancado,
            inse_municipio,
            peso_aluno,
            rede,
            {TARGET_COLUMN}
        FROM `{config.GCP_PROJECT_ID}.{config.BQ_DATASET_GOLD}.{config.ML_FEATURES_TABLE}`
        WHERE rede IN ('Municipal', 'Estadual')
    """
    log.info("Carregando dados da Gold ML no BigQuery...")
    df = client.query(query).to_dataframe()
    log.info("Dados carregados: %s registros", f"{len(df):,}")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa features (X), target binario (y) e grupo (id_aluno, usado so
    para split/CV -- nunca entra como feature do modelo).
    """
    groups = df[GROUP_COLUMN]
    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN].map({"Não": 0, "Nao": 0, "Sim": 1, "0": 0, "1": 1, 0: 0, 1: 1})
    if y.isna().any():
        raise ValueError("A variavel target possui valores nulos ou mapeamentos nao reconhecidos.")

    return X, y, groups


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_full_pipeline(model, apply_undersampling: bool = True) -> ImbPipeline:
    """
    Um unico objeto sklearn/imblearn Pipeline com pre-processamento +
    balanceamento + estimador -- atende a exigencia do PDF de "integracao
    do pre-processamento diretamente ao modelo" (na v1 do repo, o
    ColumnTransformer era ajustado fora do Pipeline do modelo).
    """
    steps = [("preprocessor", build_preprocessor())]
    if apply_undersampling:
        steps.append(("undersampler", RandomUnderSampler(random_state=config.RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def train_test_split_grouped(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    GroupShuffleSplit por id_aluno: garante que o mesmo aluno (que pode
    ter ate 2 linhas na base -- 2023 e 2024) nunca aparece em treino E
    teste ao mesmo tempo.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx], groups.iloc[test_idx],
    )


def prepare_data(client: bigquery.Client):
    df = load_data(client)
    X, y, groups = split_features_target(df)

    log.info("Distribuicao do Target: 0=%s | 1=%s", f"{(y == 0).sum():,}", f"{(y == 1).sum():,}")

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split_grouped(X, y, groups)

    overlap = set(groups_train) & set(groups_test)
    log.info(
        "Split agrupado por id_aluno -> Treino: %s | Teste: %s | Alunos em comum (deve ser 0): %d",
        f"{len(X_train):,}", f"{len(X_test):,}", len(overlap),
    )

    return X_train, X_test, y_train, y_test, groups_train, groups_test


if __name__ == "__main__":
    import google.auth
    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)
    prepare_data(client)

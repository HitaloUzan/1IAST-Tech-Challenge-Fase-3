import logging

import google.auth
from google.cloud import bigquery
import pandas as pd

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
# CONFIGURAÇÃO DAS FEATURES
# ============================================================

TARGET_COLUMN = config.TARGET_COLUMN

NUMERIC_FEATURES = [
    "ano",
    "meta_alfabetizacao_2030",
    "percentual_participacao",
    "nivel_alfabetizacao",
    "peso_aluno",
]

CATEGORICAL_FEATURES = [
    "serie",
    "rede",
    "presenca",
    "preenchimento_caderno",
    "possui_meta_municipal",
]


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def load_data(client: bigquery.Client) -> pd.DataFrame:
    """
    Carrega os dados da tabela Gold ML no BigQuery.

    A rede privada é excluída devido à baixa representatividade
    no conjunto de dados.
    """

    query = f"""
        SELECT
            ano,
            serie,
            rede,
            presenca,
            peso_aluno,
            preenchimento_caderno,
            meta_alfabetizacao_2030,
            percentual_participacao,
            nivel_alfabetizacao,
            possui_meta_municipal,
            alfabetizado
        FROM `{config.GCP_PROJECT_ID}.{config.BQ_DATASET_GOLD}.{config.ML_FEATURES_TABLE}`
        WHERE rede IN ('Municipal', 'Estadual')
    """

    log.info("Carregando dados da Gold ML...")

    df = client.query(query).to_dataframe()

    log.info(
        "Dados carregados: %s registros",
        f"{len(df):,}"
    )

    return df


# ============================================================
# SEPARAÇÃO ENTRE FEATURES E TARGET
# ============================================================

def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa as variáveis preditoras (X) da variável alvo (y).
    """

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN].map({
        "Não": 0,
        "Sim": 1,
    })

    if y.isna().any():
        raise ValueError(
            "A variável target possui valores diferentes de "
            "'Sim' e 'Não'."
        )

    return X, y


# ============================================================
# CONSTRUÇÃO DO PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Cria o pipeline de transformação das features.

    Features numéricas:
        - imputação pela mediana
        - padronização

    Features categóricas:
        - imputação pela categoria mais frequente
        - One-Hot Encoding
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def prepare_data(client: bigquery.Client):
    """
    Executa o pipeline completo de preparação dos dados.

    Fluxo:

        Gold ML
            ↓
        separação X/y
            ↓
        train/test split 80/20
            ↓
        fit do preprocessing no treino
            ↓
        transformação do treino e teste
    """

    df = load_data(client)

    X, y = split_features_target(df)

    log.info(
        "Distribuição do target:"
        " 0=%s | 1=%s",
        f"{(y == 0).sum():,}",
        f"{(y == 1).sum():,}",
    )

    # --------------------------------------------------------
    # Separação treino / teste
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    log.info(
        "Treino: %s registros | Teste: %s registros",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = build_preprocessor()

    # O fit acontece SOMENTE no conjunto de treino.
    # Isso evita data leakage.

    X_train_processed = preprocessor.fit_transform(X_train)

    # No teste apenas aplicamos as transformações
    # aprendidas no treinamento.

    X_test_processed = preprocessor.transform(X_test)

    log.info(
        "Features após preprocessing: %s",
        X_train_processed.shape[1],
    )

    log.info(
        "Shape treino: %s",
        X_train_processed.shape,
    )

    log.info(
        "Shape teste: %s",
        X_test_processed.shape,
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Executa o pipeline de preprocessing.
    """

    credentials, _ = google.auth.default()

    client = bigquery.Client(
        project=config.GCP_PROJECT_ID,
        credentials=credentials,
    )

    prepare_data(client)


if __name__ == "__main__":
    main()
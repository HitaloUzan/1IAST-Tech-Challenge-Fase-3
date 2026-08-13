import logging
from pathlib import Path

import google.auth
from google.cloud import bigquery

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.ml.preprocessing import prepare_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL_PATH = Path(config.MODEL_PATH)


# ============================================================
# TREINAMENTO
# ============================================================

def train_model(X_train, y_train):
    """
    Treina o modelo baseline de classificação.

    Logistic Regression foi escolhida como baseline por ser:
    - simples;
    - interpretável;
    - adequada para classificação binária;
    - uma referência para comparar modelos mais complexos.
    """

    log.info("Iniciando treinamento da Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    log.info("Treinamento concluído.")

    return model


# ============================================================
# AVALIAÇÃO
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo utilizando o conjunto de teste.
    """

    log.info("Iniciando avaliação...")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    log.info("===== MÉTRICAS =====")

    for metric, value in metrics.items():
        log.info(
            "%s: %.4f",
            metric.upper(),
            value,
        )

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Não alfabetizado", "Alfabetizado"],
        )
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    log.info("Iniciando pipeline de treinamento ML...")

    # --------------------------------------------------------
    # Autenticação
    # --------------------------------------------------------

    credentials, _ = google.auth.default()

    client = bigquery.Client(
        project=config.GCP_PROJECT_ID,
        credentials=credentials,
    )

    # --------------------------------------------------------
    # Preparação dos dados
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = prepare_data(client)

    # --------------------------------------------------------
    # Treinamento
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Avaliação
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Salvamento
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import joblib

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "features_numeric": [
            "ano",
            "meta_alfabetizacao_2030",
            "percentual_participacao",
            "nivel_alfabetizacao",
            "peso_aluno",
        ],
        "features_categorical": [
            "serie",
            "rede",
            "presenca",
            "preenchimento_caderno",
            "possui_meta_municipal",
        ],
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    log.info(
        "Modelo salvo em: %s",
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()
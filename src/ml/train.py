import logging
from pathlib import Path

import google.auth
from google.cloud import bigquery
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

RANDOM_STATE = config.RANDOM_STATE


# ============================================================
# MODELOS
# ============================================================

def build_model(model_name):
    """
    Cria o modelo de acordo com o experimento.
    """

    if model_name == "logistic_baseline":

        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "logistic_balanced":

        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

    if model_name == "random_forest_balanced":

        return RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

    raise ValueError(
        f"Modelo não reconhecido: {model_name}"
    )


# ============================================================
# TREINAMENTO
# ============================================================

def train_model(model, X_train, y_train):
    """
    Treina o modelo recebido.
    """

    log.info(
        "Iniciando treinamento: %s",
        model.__class__.__name__,
    )

    model.fit(
        X_train,
        y_train,
    )

    log.info("Treinamento concluído.")

    return model


# ============================================================
# AVALIAÇÃO
# ============================================================

def evaluate_model(
    model,
    model_name,
    X_test,
    y_test,
):
    """
    Avalia o modelo e retorna suas métricas.
    """

    log.info(
        "Iniciando avaliação do modelo: %s",
        model_name,
    )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
        ),
        "recall": recall_score(
            y_test,
            predictions,
        ),
        "f1": f1_score(
            y_test,
            predictions,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    log.info("===== MÉTRICAS =====")

    for metric, value in metrics.items():

        if metric != "model":

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
            target_names=[
                "Não alfabetizado",
                "Alfabetizado",
            ],
        )
    )

    return metrics


# ============================================================
# SALVAMENTO
# ============================================================

def save_model(
    model,
    model_name,
    preprocessor,
    metrics,
):
    """
    Salva o modelo e o preprocessing juntos.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODELS_DIR
        / f"{model_name}.joblib"
    )

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
        model_path,
    )

    log.info(
        "Modelo salvo em: %s",
        model_path,
    )


# ============================================================
# RESULTADOS
# ============================================================

def save_results(results):
    """
    Salva os resultados dos experimentos em CSV.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        REPORTS_DIR
        / "model_results.csv"
    )

    df_results = pd.DataFrame(results)

    df_results.to_csv(
        results_path,
        index=False,
    )

    log.info(
        "Resultados salvos em: %s",
        results_path,
    )

    print("\n===== COMPARAÇÃO DOS MODELOS =====")

    print(
        df_results[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    log.info(
        "Iniciando pipeline de treinamento ML..."
    )

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

    log.info(
        "Dados preparados para os experimentos."
    )

    # --------------------------------------------------------
    # Experimentos
    # --------------------------------------------------------

    experiments = [
        "random_forest_balanced",
    ]

    results = []

    for model_name in experiments:

        log.info(
            "========================================"
        )

        log.info(
            "EXPERIMENTO: %s",
            model_name,
        )

        log.info(
            "========================================"
        )

        # Criar modelo
        model = build_model(
            model_name
        )

        # Treinar
        model = train_model(
            model,
            X_train,
            y_train,
        )

        # Avaliar
        metrics = evaluate_model(
            model,
            model_name,
            X_test,
            y_test,
        )

        # Salvar modelo
        save_model(
            model,
            model_name,
            preprocessor,
            metrics,
        )

        results.append(
            metrics
        )

    # --------------------------------------------------------
    # Salvar resultados
    # --------------------------------------------------------

    save_results(
        results
    )

    log.info(
        "Pipeline de treinamento concluída."
    )


if __name__ == "__main__":
    main()
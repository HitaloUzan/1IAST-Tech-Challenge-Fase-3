import logging
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def load_pipeline(model_filename: str = "xgboost.joblib"):
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em: {model_path}")
    log.info("Carregando pipeline: %s", model_path)
    return joblib.load(model_path)


def predict_risk(
    df_input: pd.DataFrame,
    model_filename: str = "xgboost.joblib",
    threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Recebe dados brutos (mesmas colunas da Gold ML) e aplica o pipeline
    completo (preprocessor + modelo em um unico objeto) direto -- nao ha
    mais preprocessor separado pra carregar.
    """
    pipeline = load_pipeline(model_filename)

    log.info("Calculando probabilidades de alfabetizacao...")
    probabilities_alfabetizado = pipeline.predict_proba(df_input)[:, 1]

    df_results = df_input.copy()
    df_results["probabilidade_alfabetizado"] = probabilities_alfabetizado.round(4)
    df_results["predicao_final"] = (probabilities_alfabetizado >= threshold).astype(int)
    df_results["status_risco"] = df_results["predicao_final"].map({
        0: "ALTO RISCO (Busca Ativa)",
        1: "Baixo Risco / Adequado",
    })

    log.info("Inferencia concluida com sucesso!")
    return df_results


def main() -> None:
    sample_data = pd.DataFrame([
        {
            "taxa_alfabetizacao_municipio": 82.5,
            "media_portugues_municipio": 195.4,
            "proporcao_abaixo_basico": 0.35,
            "proporcao_basico": 0.40,
            "proporcao_adequado_avancado": 0.25,
            "inse_municipio": 4.2,
            "peso_aluno": 1.0,
            "rede": "Municipal",
        },
        {
            "taxa_alfabetizacao_municipio": 60.1,
            "media_portugues_municipio": 160.0,
            "proporcao_abaixo_basico": 0.60,
            "proporcao_basico": 0.30,
            "proporcao_adequado_avancado": 0.10,
            "inse_municipio": 2.8,
            "peso_aluno": 1.0,
            "rede": "Estadual",
        },
    ])

    log.info("Executando inferencia de teste...")
    results = predict_risk(sample_data, model_filename="xgboost.joblib", threshold=0.55)

    print("\n===== RESULTADOS DA PREDICAO (INFERENCIA) =====")
    print(results[["rede", "inse_municipio", "probabilidade_alfabetizado", "status_risco"]])


if __name__ == "__main__":
    main()

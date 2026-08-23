import logging
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def load_trained_artifact(model_filename: str = "xgboost.joblib"):
    """
    Carrega o artefato salvo contendo o modelo e o preprocessor.
    """
    model_path = MODELS_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

    log.info("Carregando artefato do modelo: %s", model_path)
    artifact = joblib.load(model_path)
    return artifact


def predict_risk(
    df_input: pd.DataFrame,
    model_filename: str = "xgboost.joblib",
    threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Aplica o preprocessor nos dados brutos de entrada, realiza a predição de risco
    e aplica o limiar de negócio (0.55).
    """
    artifact = load_trained_artifact(model_filename)
    
    # Se o artefato for um dicionário contendo modelo + preprocessor separados
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        preprocessor = artifact.get("preprocessor", None)
    else:
        model = artifact
        preprocessor = None

    # Se a entrada ainda tiver strings categóricas, converte para category
    df_processed = df_input.copy()
    if "rede" in df_processed.columns:
        df_processed["rede"] = df_processed["rede"].astype("category")

    # Aplica o preprocessor caso ele exista
    if preprocessor is not None:
        X_trans = preprocessor.transform(df_processed)
    else:
        X_trans = df_processed

    log.info("Calculando probabilidades de alfabetização...")
    probabilities_alfabetizado = model.predict_proba(X_trans)[:, 1]

    df_results = df_input.copy()
    df_results["probabilidade_alfabetizado"] = probabilities_alfabetizado.round(4)

    # Regra de Negócio com Threshold Tuning
    df_results["predicao_final"] = (probabilities_alfabetizado >= threshold).astype(int)
    df_results["status_risco"] = df_results["predicao_final"].map({
        0: "ALTO RISCO (Busca Ativa)",
        1: "Baixo Risco / Adequado",
    })

    log.info("Inferência concluída com sucesso!")
    return df_results


def main() -> None:
    # Exemplo sintético simulando registros brutos da camada Gold
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

    log.info("Executando inferência de teste...")
    results = predict_risk(sample_data, model_filename="xgboost.joblib", threshold=0.55)

    print("\n===== RESULTADOS DA PREDIÇÃO (INFERÊNCIA) =====")
    print(results[["rede", "inse_municipio", "probabilidade_alfabetizado", "status_risco"]])


if __name__ == "__main__":
    main()
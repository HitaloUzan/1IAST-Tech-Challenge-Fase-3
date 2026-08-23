import logging
from pathlib import Path

import google.auth
from google.cloud import bigquery
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config
from src.ml.preprocessing import prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
IMAGES_DIR = Path("reports/images")


def generate_shap_and_importance(model_filename: str = "xgboost.joblib"):
    """
    Gera o Feature Importance (porcentagens) e as 3 visualizações SHAP
    baseando-se nas variáveis atuais da camada Gold.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

    log.info("Carregando artefato: %s", model_path)
    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact

    # 1. Obter dados processados para calcular o SHAP
    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)
    
    # Pegamos uma amostra do conjunto de teste
    _, X_test, _, _, preprocessor = prepare_data(client, apply_undersampling=False)

    # Nomes reais das colunas pós-preprocessor
    feature_names = [
        "taxa_alfabetizacao_municipio",
        "media_portugues_municipio",
        "proporcao_abaixo_basico",
        "proporcao_basico",
        "proporcao_adequado_avancado",
        "inse_municipio",
        "peso_aluno",
        "rede_Estadual",
        "rede_Municipal",
    ]

    # ------------------------------------------------------------
    # A. TABELA DE CONTRIBUIÇÃO (PORCENTAGEM POR VARIÁVEL)
    # ------------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pct_importances = (importances / importances.sum()) * 100

        df_imp = pd.DataFrame({
            "Variável": feature_names[:len(importances)],
            "Ganho (Gain)": importances,
            "Contribuição (%)": np.round(pct_importances, 2)
        }).sort_values(by="Contribuição (%)", ascending=False)

        log.info("\n===== TABELA DE CONTRIBUIÇÃO DAS VARIÁVEIS =====\n%s", df_imp.to_string(index=False))

        # 1. Gráfico: Quais variáveis mais impactam a Alfabetização Infantil? (XGBoost Gain)
        plt.figure(figsize=(10, 6))
        df_imp_plot = df_imp.sort_values(by="Contribuição (%)", ascending=True)
        plt.barh(df_imp_plot["Variável"], df_imp_plot["Gain Importância" if "Gain Importância" in df_imp_plot else "Ganho (Gain)"], color="#008080")
        plt.title("Quais variáveis mais impactam a Alfabetização Infantil? (XGBoost)", fontsize=12, fontweight="bold")
        plt.xlabel("Importância Relativa (Gain)")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "feature_importance_xgboost.png", dpi=300)
        plt.close()

    # ------------------------------------------------------------
    # B. CÁLCULO E GRÁFICOS DO SHAP
    # ------------------------------------------------------------
    log.info("Calculando valores SHAP (amostra de 2.000 registros)...")
    sample_size = min(2000, X_test.shape[0])
    X_sample = X_test[:sample_size]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Atribui os nomes das colunas ao objeto SHAP
    shap_values.feature_names = feature_names[:X_sample.shape[1]]

    # 2. Gráfico: Impacto Direcional das Variáveis na Alfabetização Infantil (SHAP Summary)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Impacto Direcional das Variáveis na Alfabetização Infantil (SHAP)", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "shap_summary_plot.png", dpi=300)
    plt.close()

    # 3. Gráfico: SHAP Waterfall (Exemplo de um aluno individual)
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Por que este aluno específico foi classificado assim? (SHAP Waterfall)", fontsize=11, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "shap_waterfall_single.png", dpi=300)
    plt.close()

    log.info("Todos os 3 gráficos SHAP e a tabela de porcentagens foram salvos em: %s", IMAGES_DIR)


if __name__ == "__main__":
    generate_shap_and_importance()
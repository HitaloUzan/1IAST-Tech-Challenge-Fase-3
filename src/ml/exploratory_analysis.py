import logging
from pathlib import Path

import google.auth
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from google.cloud import bigquery

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IMAGES_DIR = Path("reports/images")

NUMERIC_FEATURES = [
    "taxa_alfabetizacao_municipio",
    "media_portugues_municipio",
    "proporcao_abaixo_basico",
    "proporcao_basico",
    "proporcao_adequado_avancado",
    "inse_municipio",
    "peso_aluno",
]


def load_gold_data(client: bigquery.Client) -> pd.DataFrame:
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
    log.info("Carregando dados para Analise Exploratoria...")
    return client.query(query).to_dataframe()


def run_exploratory_analysis(df: pd.DataFrame) -> dict:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    log.info("Gerando estatisticas descritivas...")
    stats = {
        "shape": df.shape,
        "null_count": df.isna().sum().to_dict(),
        "target_distribution": df["alfabetizado"].value_counts(normalize=True).to_dict(),
        "describe_numeric": df[NUMERIC_FEATURES].describe().to_dict(),
    }

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="alfabetizado", hue="alfabetizado", palette="Blues_r", legend=False)
    plt.title("Distribuicao da Variavel Alvo (Alfabetizado)")
    plt.xlabel("Status de Alfabetizacao")
    plt.ylabel("Quantidade de registros aluno x edicao")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "target_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    df_corr = df[NUMERIC_FEATURES].copy()
    df_corr["target_num"] = df["alfabetizado"].map({"Não": 0, "Nao": 0, "Sim": 1, 0: 0, 1: 1})
    corr_matrix = df_corr.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title("Matriz de Correlacao - Camada Gold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_matrix.png", dpi=300)
    plt.close()

    if "inse_municipio" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="alfabetizado", y="inse_municipio", hue="alfabetizado", palette="Set2", legend=False)
        plt.title("Distribuicao do INSE Municipal por Status de Alfabetizacao")
        plt.xlabel("Alfabetizado")
        plt.ylabel("INSE do Municipio")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "inse_vs_target.png", dpi=300)
        plt.close()

    log.info("Analise exploratoria concluida! Graficos salvos em: %s", IMAGES_DIR)
    return stats


def main() -> None:
    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)
    df = load_gold_data(client)
    run_exploratory_analysis(df)


if __name__ == "__main__":
    main()

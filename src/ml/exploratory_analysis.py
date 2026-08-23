import logging
from pathlib import Path

import google.auth
from google.cloud import bigquery
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
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
    """Carrega os dados completos da camada Gold para análise."""
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
    log.info("Carregando dados para Análise Exploratória...")
    return client.query(query).to_dataframe()


def run_exploratory_analysis(df: pd.DataFrame) -> dict:
    """
    Executa a análise estatística descritiva e gera os gráficos salvando em reports/images/.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    log.info("Gerando estatísticas descritivas...")
    stats = {
        "shape": df.shape,
        "null_count": df.isna().sum().to_dict(),
        "target_distribution": df["alfabetizado"].value_counts(normalize=True).to_dict(),
        "describe_numeric": df[NUMERIC_FEATURES].describe().to_dict(),
    }

    # 1. Gráfico de Distribuição da Variável Alvo
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="alfabetizado", palette="Blues_r")
    plt.title("Distribuição da Variável Alvo (Alfabetizado)")
    plt.xlabel("Status de Alfabetização")
    plt.ylabel("Quantidade de Alunos")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "target_distribution.png", dpi=300)
    plt.close()

    # 2. Matriz de Correlação das Variáveis Numéricas
    plt.figure(figsize=(10, 8))
    # Mapeia temporariamente a target para cálculo de correlação
    df_corr = df[NUMERIC_FEATURES].copy()
    df_corr["target_num"] = df["alfabetizado"].map({"Não": 0, "Sim": 1, 0: 0, 1: 1})
    
    corr_matrix = df_corr.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title("Matriz de Correlação - Camada Gold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_matrix.png", dpi=300)
    plt.close()

    # 3. Distribuição do INSE por Status de Alfabetização
    if "inse_municipio" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="alfabetizado", y="inse_municipio", palette="Set2")
        plt.title("Distribuição do INSE Municipal por Status de Alfabetização")
        plt.xlabel("Alfabetizado")
        plt.ylabel("INSE do Município")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "inse_vs_target.png", dpi=300)
        plt.close()

    log.info("Análise exploratória concluída! Gráficos salvos em: %s", IMAGES_DIR)
    return stats


def main() -> None:
    credentials, _ = google.auth.default()
    client = bigquery.Client(
        project=config.GCP_PROJECT_ID,
        credentials=credentials,
    )
    df = load_gold_data(client)
    run_exploratory_analysis(df)


if __name__ == "__main__":
    main()
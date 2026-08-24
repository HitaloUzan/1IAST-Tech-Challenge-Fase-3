import sys
import logging

from google.cloud import bigquery
import google.auth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, ".")
import config

GCP_PROJECT_ID = config.GCP_PROJECT_ID
BQ_DATASET_GOLD = config.BQ_DATASET_GOLD
BQ_DATASET_SILVER = config.BQ_DATASET_SILVER
ML_FEATURES_TABLE = config.ML_FEATURES_TABLE

PARTITION_BY_ANO = {ML_FEATURES_TABLE}
CLUSTER_FIELDS = {
    ML_FEATURES_TABLE: ["id_municipio", "rede"],
}

# Query herdada de NaiaraMartins/1IAST-Tech-Challenge-Fase-3 (src/gold/build_gold_ml.py),
# com id_aluno preservado para permitir split/CV agrupado por aluno (evita o mesmo
# aluno aparecer em treino e teste quando ele tem mais de um caderno na mesma edicao),
# mais 3 melhorias sobre a v2:
#
# 1. municipio_same_year: a v2 fazia AVG(taxa_alfabetizacao) misturando 2023+2024
#    pra toda linha (sem filtrar por ano) -- ou seja, uma linha de 2023 recebia
#    informacao de 2024 (que ainda nao existia). Corrigido para casar pelo MESMO
#    ano da linha.
# 2. escola_prior_year: feature nova, no grao de ESCOLA (mais fino que municipio),
#    calculada a partir do proprio alunos_clean agregado por id_escola+ano e casada
#    com ano_alvo = ano+1 -- ou seja, so usa o desempenho da escola no ano ANTERIOR
#    (nunca o mesmo ano/mesma turma), sem risco de vazamento. Cobertura: ~79% das
#    linhas de 2024 tem historico de 2023; linhas de 2023 ficam NULL (nao ha 2022
#    nos dados) -- sinalizado explicitamente por tem_historico_escola.
#    (Tentamos ligar INSE por escola tambem, mas id_escola em inse_escola_clean
#    e silver.alunos_clean nao compartilham o mesmo namespace de codigo -- 0 de
#    42.802 escolas batem mesmo apos normalizar o tipo. Confirmado por query
#    direta antes de descartar a ideia.)
# 3. sigla_uf_code: os 2 primeiros digitos do id_municipio (codigo IBGE) --
#    variavel territorial adicional, praticamente gratuita.
ML_GOLD_TABLES = {
    ML_FEATURES_TABLE: f"""
        WITH municipio_same_year AS (
            SELECT
                SUBSTR(CAST(id_municipio AS STRING), 1, 6) AS id_municipio_6dig,
                CAST(ano AS INT64) AS ano,
                AVG(SAFE_CAST(REPLACE(CAST(taxa_alfabetizacao AS STRING), ',', '.') AS FLOAT64)) AS taxa_alfabetizacao_municipio,
                AVG(SAFE_CAST(REPLACE(CAST(media_portugues AS STRING), ',', '.') AS FLOAT64)) AS media_portugues_municipio,
                AVG(SAFE_CAST(REPLACE(CAST(proporcao_abaixo_basico AS STRING), ',', '.') AS FLOAT64)) AS proporcao_abaixo_basico,
                AVG(SAFE_CAST(REPLACE(CAST(proporcao_basico AS STRING), ',', '.') AS FLOAT64)) AS proporcao_basico,
                AVG(SAFE_CAST(REPLACE(CAST(proporcao_adequado_avancado AS STRING), ',', '.') AS FLOAT64)) AS proporcao_adequado_avancado
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.alfabetizacao_municipio_clean`
            WHERE TRIM(serie) = '2° ano do Ensino Fundamental'
            GROUP BY 1, 2
        ),
        escola_prior_year AS (
            SELECT
                TRIM(CAST(SAFE_CAST(id_escola AS INT64) AS STRING)) AS id_escola,
                CAST(ano AS INT64) + 1 AS ano_alvo,
                AVG(CASE WHEN alfabetizado = 'Sim' THEN 1.0 ELSE 0.0 END) AS taxa_alfabetizacao_escola_prior,
                COUNT(*) AS n_alunos_prior_escola
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.alunos_clean`
            WHERE presenca = 'Presente'
            GROUP BY 1, 2
        ),
        inse_municipio_agregado AS (
            SELECT
                SUBSTR(CAST(id_municipio AS STRING), 1, 6) AS id_municipio_6dig,
                AVG(SAFE_CAST(REPLACE(CAST(inse AS STRING), ',', '.') AS FLOAT64)) AS inse_municipio
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.inse_escola_clean`
            WHERE id_municipio IS NOT NULL
            GROUP BY 1
        )
        SELECT
            CAST(a.ano AS INT64) AS ano,
            a.id_aluno,
            SUBSTR(CAST(a.id_municipio AS STRING), 1, 6) AS id_municipio,
            SUBSTR(CAST(a.id_municipio AS STRING), 1, 2) AS sigla_uf_code,
            TRIM(CAST(SAFE_CAST(a.id_escola AS INT64) AS STRING)) AS id_escola,
            TRIM(a.rede) AS rede,
            a.presenca,
            SAFE_CAST(REPLACE(CAST(a.peso_aluno AS STRING), ',', '.') AS FLOAT64) AS peso_aluno,
            a.alfabetizado,

            i.inse_municipio,

            m.taxa_alfabetizacao_municipio,
            m.media_portugues_municipio,
            m.proporcao_abaixo_basico,
            m.proporcao_basico,
            m.proporcao_adequado_avancado,

            e.taxa_alfabetizacao_escola_prior,
            e.n_alunos_prior_escola,
            CASE WHEN e.id_escola IS NOT NULL THEN 1 ELSE 0 END AS tem_historico_escola

        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.alunos_clean` a

        LEFT JOIN inse_municipio_agregado i
            ON SUBSTR(CAST(a.id_municipio AS STRING), 1, 6) = i.id_municipio_6dig

        LEFT JOIN municipio_same_year m
            ON SUBSTR(CAST(a.id_municipio AS STRING), 1, 6) = m.id_municipio_6dig
            AND CAST(a.ano AS INT64) = m.ano

        LEFT JOIN escola_prior_year e
            ON TRIM(CAST(SAFE_CAST(a.id_escola AS INT64) AS STRING)) = e.id_escola
            AND CAST(a.ano AS INT64) = e.ano_alvo

        WHERE a.id_municipio IS NOT NULL
          AND a.id_aluno IS NOT NULL
    """,
}


def ensure_dataset(client: bigquery.Client, dataset_id: str) -> None:
    dataset_ref = bigquery.Dataset(f"{GCP_PROJECT_ID}.{dataset_id}")
    dataset_ref.location = "US"
    client.create_dataset(dataset_ref, exists_ok=True)
    log.info(f"Dataset gold '{dataset_id}' verificado ou criado com sucesso.")


def build_table(client: bigquery.Client, table_name: str, query: str) -> int:
    destination = f"{GCP_PROJECT_ID}.{BQ_DATASET_GOLD}.{table_name}"

    job_config = bigquery.QueryJobConfig(
        destination=destination,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )

    if table_name in PARTITION_BY_ANO:
        job_config.range_partitioning = bigquery.RangePartitioning(
            field="ano",
            range_=bigquery.PartitionRange(start=2019, end=2051, interval=1),
        )

    if table_name in CLUSTER_FIELDS:
        job_config.clustering_fields = CLUSTER_FIELDS[table_name]

    log.info(f"Executando query de compilacao da tabela gold.{table_name}...")
    job = client.query(query, job_config=job_config)
    job.result()

    table = client.get_table(destination)
    log.info(f"Tabela gold.{table_name} processada! Contem {table.num_rows:,} registros.")
    return table.num_rows


def main() -> None:
    log.info("Iniciando execucao da Camada Gold ML (versao com id_aluno)...")
    credentials, _ = google.auth.default()
    client = bigquery.Client(project=GCP_PROJECT_ID, credentials=credentials)

    ensure_dataset(client, BQ_DATASET_GOLD)

    for table_name, query in ML_GOLD_TABLES.items():
        build_table(client, table_name, query)

    log.info("Processamento da camada Gold ML concluido com sucesso!")


if __name__ == "__main__":
    main()

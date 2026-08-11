import sys
import logging
from datetime import datetime

import google.auth
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log = logging.getLogger(__name__)

sys.path.insert(0, ".")

import config

GCP_PROJECT_ID = config.GCP_PROJECT_ID
BQ_DATASET_GOLD = config.BQ_DATASET_GOLD
BQ_DATASET_SILVER = config.BQ_DATASET_SILVER
ML_FEATURES_TABLE = config.ML_FEATURES_TABLE

ML_GOLD_TABLES = {

    ML_FEATURES_TABLE: f"""

        WITH alunos AS (

            SELECT
                ano,
                id_municipio,
                serie,
                rede,
                alfabetizado

            FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.alunos_clean`

        ),

        metas AS (

            SELECT
                ano,
                id_municipio,
                rede,
                meta_alfabetizacao_2030,
                percentual_participacao,
                nivel_alfabetizacao

            FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.metas_consolidadas`

            WHERE escopo = 'municipio'

        )

        SELECT

            a.ano,

            a.id_municipio,

            a.serie,

            a.rede,

            m.meta_alfabetizacao_2030,

            m.percentual_participacao,

            m.nivel_alfabetizacao,

            CASE
                WHEN m.id_municipio IS NOT NULL THEN TRUE
                ELSE FALSE
            END AS possui_meta_municipal,

            a.alfabetizado

        FROM alunos a

        LEFT JOIN metas m
            ON a.ano = m.ano
           AND a.id_municipio = m.id_municipio
           AND a.rede = m.rede

    """
     }
def ensure_dataset(client: bigquery.Client, dataset_id: str) -> None:
    dataset_ref = bigquery.Dataset(f"{GCP_PROJECT_ID}.{dataset_id}")
    dataset_ref.location = "US"
    client.create_dataset(dataset_ref, exists_ok=True)


def build_table(client: bigquery.Client, table_name: str, query: str) -> int:

    destination = f"{GCP_PROJECT_ID}.{BQ_DATASET_GOLD}.{table_name}"

    job_config = bigquery.QueryJobConfig(
        destination=destination,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )

    log.info(f"Building gold.{table_name} ...")

    job = client.query(query, job_config=job_config)
    job.result()

    table = client.get_table(destination)

    log.info(f"gold.{table_name} -> {table.num_rows:,} rows")

    return table.num_rows


def main() -> None:
    
    credentials, _ = google.auth.default()

    client = bigquery.Client(
        project=GCP_PROJECT_ID,
        credentials=credentials
    )

    ensure_dataset(client, BQ_DATASET_GOLD)

    start = datetime.now()

    results = {}
    errors = []

    for table_name, query in ML_GOLD_TABLES.items():

        try:
            rows = build_table(client, table_name, query)
            results[table_name] = rows

        except Exception as exc:
            log.error(f"Failed to build gold.{table_name}: {exc}")
            errors.append(table_name)

    elapsed = (datetime.now() - start).total_seconds()

    log.info(
        f"Gold ML build complete in {elapsed:.1f}s | "
        f"success={len(results)} error={len(errors)}"
    )

    if errors:
        log.error(f"Tables with errors: {errors}")
        sys.exit(1)


if __name__ == "__main__":
      main()
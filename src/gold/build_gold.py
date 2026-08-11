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

from config import (
    GCP_PROJECT_ID,
    BQ_DATASET_GOLD,
    BQ_DATASET_SILVER,
    ML_FEATURES_TABLE
)
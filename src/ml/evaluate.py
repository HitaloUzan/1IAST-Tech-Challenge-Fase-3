import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

def run_threshold_simulation(y_test, y_proba, thresholds=[0.50, 0.52, 0.55, 0.58, 0.60]):
    """
    Simula o impacto de diferentes limiares de corte na detecção de alunos vulneráveis.
    """
    results = []
    
    for t in thresholds:
        # Probabilidade de ser alfabetizado < t -> Classifica como Não Alfabetizado (0)
        y_pred = (y_proba >= t).astype(int)
        
        rec_0 = recall_score(y_test, y_pred, pos_label=0)
        prec_0 = precision_score(y_test, y_pred, pos_label=0)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            "threshold": t,
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_mac, 4),
            "recall_nao_alfabetizado": round(rec_0, 4),
            "precision_nao_alfabetizado": round(prec_0, 4)
        })
        
    df_sim = pd.DataFrame(results)
    return df_sim

def evaluate_saved_model(model_filename="xgboost.joblib", threshold=0.55):
    """
    Carrega o modelo salvo e avalia com a regra de negócio do limiar de 0.55.
    """
    model_path = MODELS_DIR / model_filename
    
    if not model_path.exists():
        log.error("Modelo não encontrado em: %s", model_path)
        return
        
    log.info("Carregando artefato do modelo: %s", model_path)
    artifact = joblib.load(model_path)
    
    model = artifact["model"]
    metrics_baseline = artifact.get("metrics", {})
    
    log.info("Métricas Baseline do Treinamento (Threshold 0.50): %s", metrics_baseline)
    
    # Exemplo de chamada para simulação de limiar caso os dados de teste sejam passados
    # y_pred_opt = (y_proba >= threshold).astype(int)
    
    return artifact

if __name__ == "__main__":
    log.info("Executando módulo de avaliação e Threshold Tuning...")
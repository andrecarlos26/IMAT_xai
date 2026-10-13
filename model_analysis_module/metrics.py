import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_metrics(y_true, y_pred):
    """
    Calculates MAE, MSE, RMSE and R² from the y_true and y_pred arrays.
    Returns a dict with the metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}


def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5):
    """
    Calcula métricas de classificação binária dado vetor de probabilidades (y_prob).
    Retorna accuracy, precision, recall, f1 e auc (quando possível).
    """
    y_true = np.array(y_true).astype(int).ravel()
    y_prob = np.array(y_prob).astype(float).ravel()
    y_pred = (y_prob >= float(threshold)).astype(int)

    out = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0)
    }
    # AUC exige ao menos duas classes presentes
    try:
        out['auc'] = roc_auc_score(y_true, y_prob)
    except Exception:
        out['auc'] = float('nan')
    return out

"""Reusable explainability utilities for IMAT.

The functions in this module are intentionally defensive: every diagnostic tries to
produce a useful result without breaking model training when an optional
explainability library is missing or when a dataset is too small.
"""
from __future__ import annotations

import base64
import io
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.calibration import calibration_curve

try:  # SHAP is useful, but should never be a hard dependency at runtime.
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    shap = None

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - imported by CNN/MLP environments
    tf = None


MAX_EXPLAINABILITY_ROWS = 60
MAX_FEATURES_ON_CHART = 12
EPS = 1e-12


@dataclass(frozen=True)
class ExplainabilityArtifact:
    key: str
    title: str
    description: str
    image: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    kind: str = "diagnostic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "image": self.image,
            "summary": _json_safe(self.summary or {}),
            "kind": self.kind,
        }


def _labels(language: str) -> Dict[str, str]:
    lang = "en" if (language or "").lower().startswith("en") else "pt"
    if lang == "en":
        return {
            "training_loss": "Training loss",
            "validation_loss": "Validation loss",
            "epoch": "Epoch",
            "loss": "Loss",
            "tree": "Number of trees",
            "train_mse": "Training MSE",
            "test_mse": "Test MSE",
            "feature_importance": "Feature importance",
            "importance": "Importance",
            "feature": "Feature",
            "confusion_matrix": "Confusion matrix",
            "predicted": "Predicted",
            "actual": "Actual",
            "reliability": "Reliability diagram",
            "mean_predicted": "Mean predicted value",
            "observed": "Observed value",
            "perfect": "Perfect calibration",
            "probability": "Predicted probability",
            "fraction": "Fraction of positives",
            "gradcam": "Grad-CAM / saliency",
            "image_shap": "SHAP Implementation for Image Data",
            "token_position": "Token / position",
            "score": "Score",
            "fallback": "Fallback method",
            "not_applicable": "Not applicable for this task",
        }
    return {
        "training_loss": "Loss de treino",
        "validation_loss": "Loss de validação",
        "epoch": "Época",
        "loss": "Loss",
        "tree": "Número de árvores",
        "train_mse": "MSE de treino",
        "test_mse": "MSE de teste",
        "feature_importance": "Importância das features",
        "importance": "Importância",
        "feature": "Feature",
        "confusion_matrix": "Matriz de confusão",
        "predicted": "Predito",
        "actual": "Real",
        "reliability": "Diagrama de confiabilidade",
        "mean_predicted": "Valor médio predito",
        "observed": "Valor observado",
        "perfect": "Calibração perfeita",
        "probability": "Probabilidade predita",
        "fraction": "Fração de positivos",
        "gradcam": "Grad-CAM / saliência",
        "image_shap": "SHAP Implementation for Image Data",
        "token_position": "Token / posição",
        "score": "Pontuação",
        "fallback": "Método alternativo",
        "not_applicable": "Não aplicável para esta tarefa",
    }


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _encode_figure(fig) -> str:
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _safe_sample(X: np.ndarray, max_rows: int = MAX_EXPLAINABILITY_ROWS) -> np.ndarray:
    X = np.asarray(X)
    if X.shape[0] <= max_rows:
        return X
    idx = np.linspace(0, X.shape[0] - 1, max_rows).astype(int)
    return X[idx]


def _safe_feature_names(feature_names: Sequence[str], n_features: int) -> List[str]:
    names = [str(v) for v in feature_names]
    if len(names) < n_features:
        names.extend([f"feature_{i}" for i in range(len(names), n_features)])
    return names[:n_features]


def _bar_chart(title: str, names: Sequence[str], values: Sequence[float], labels: Dict[str, str]) -> str:
    names = list(names)
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)[::-1]
    order = order[:MAX_FEATURES_ON_CHART]
    plot_names = [names[i] for i in order][::-1]
    plot_values = values[order][::-1]

    fig, ax = plt.subplots(figsize=(8, max(3.8, len(plot_names) * 0.38)))
    ax.barh(plot_names, plot_values)
    ax.set_title(title)
    ax.set_xlabel(labels["importance"])
    ax.set_ylabel(labels["feature"])
    return _encode_figure(fig)


def _heatmap(title: str, matrix: np.ndarray, xlabels: Sequence[str], ylabels: Sequence[str], labels: Dict[str, str]) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(labels["predicted"])
    ax.set_ylabel(labels["actual"])
    ax.set_xticks(range(len(xlabels)), labels=xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels)), labels=ylabels)

    max_value = float(np.max(matrix)) if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{int(value)}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _encode_figure(fig)


def history_loss_curve(history: Any, language: str = "pt") -> ExplainabilityArtifact:
    labels = _labels(language)
    hist = getattr(history, "history", history) or {}
    loss = np.asarray(hist.get("loss", []), dtype=float)
    val_loss = np.asarray(hist.get("val_loss", []), dtype=float)

    if loss.size == 0:
        return ExplainabilityArtifact(
            key="loss_curve",
            title=labels["training_loss"],
            description=labels["not_applicable"],
            summary={"available": False},
            kind="training_curve",
        )

    epochs = np.arange(1, loss.size + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(epochs, loss, marker="o", label=labels["training_loss"])
    if val_loss.size:
        ax.plot(np.arange(1, val_loss.size + 1), val_loss, marker="o", label=labels["validation_loss"])
    ax.set_title(f"{labels['training_loss']} vs. {labels['validation_loss']}")
    ax.set_xlabel(labels["epoch"])
    ax.set_ylabel(labels["loss"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_source = val_loss if val_loss.size else loss
    best_epoch = int(np.nanargmin(best_source)) + 1
    final_gap = None
    if val_loss.size and loss.size:
        final_gap = float(val_loss[-1] - loss[-1])

    return ExplainabilityArtifact(
        key="loss_curve",
        title=f"{labels['training_loss']} vs. {labels['validation_loss']}",
        description=(
            "Mostra se o erro de treino e validação caminham juntos ou se há sinais de overfitting."
            if not (language or "").lower().startswith("en")
            else "Shows whether training and validation errors move together or whether there are overfitting signals."
        ),
        image=_encode_figure(fig),
        summary={
            "available": True,
            "best_epoch": best_epoch,
            "final_train_loss": float(loss[-1]),
            "final_validation_loss": float(val_loss[-1]) if val_loss.size else None,
            "final_validation_train_gap": final_gap,
        },
        kind="training_curve",
    )


def random_forest_tree_curve(
    rf_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    language: str = "pt",
) -> ExplainabilityArtifact:
    labels = _labels(language)
    estimators = list(getattr(rf_model, "estimators_", []))
    if not estimators:
        return ExplainabilityArtifact(
            key="rf_tree_curve",
            title=f"{labels['train_mse']} vs. {labels['test_mse']}",
            description=labels["not_applicable"],
            summary={"available": False},
            kind="training_curve",
        )

    max_points = min(len(estimators), 80)
    step = max(1, len(estimators) // max_points)
    train_sum = np.zeros(len(X_train), dtype=float)
    test_sum = np.zeros(len(X_test), dtype=float)
    train_mse: List[float] = []
    test_mse: List[float] = []
    counts: List[int] = []

    for i, estimator in enumerate(estimators, start=1):
        train_sum += estimator.predict(X_train)
        test_sum += estimator.predict(X_test)
        if i == 1 or i == len(estimators) or i % step == 0:
            counts.append(i)
            train_mse.append(float(mean_squared_error(y_train, train_sum / i)))
            test_mse.append(float(mean_squared_error(y_test, test_sum / i)))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(counts, train_mse, marker="o", label=labels["train_mse"])
    ax.plot(counts, test_mse, marker="o", label=labels["test_mse"])
    ax.set_title(f"{labels['train_mse']} vs. {labels['test_mse']}")
    ax.set_xlabel(labels["tree"])
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_idx = int(np.nanargmin(test_mse))
    return ExplainabilityArtifact(
        key="rf_tree_curve",
        title=f"{labels['train_mse']} vs. {labels['test_mse']}",
        description=(
            "Acompanha o erro à medida que novas árvores entram no ensemble, substituindo a curva de loss em modelos sem épocas."
            if not (language or "").lower().startswith("en")
            else "Tracks error as trees are added to the ensemble, replacing the loss curve for models without epochs."
        ),
        image=_encode_figure(fig),
        summary={
            "available": True,
            "best_tree_count": counts[best_idx],
            "best_test_mse": test_mse[best_idx],
            "final_train_mse": train_mse[-1],
            "final_test_mse": test_mse[-1],
        },
        kind="training_curve",
    )


def _permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    seed: Optional[int],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    baseline_pred = np.asarray(predict_fn(X)).ravel()
    if task == "binary":
        baseline_score = np.mean((baseline_pred >= 0.5).astype(int) == y.astype(int))
    else:
        baseline_score = -mean_squared_error(y, baseline_pred)

    importances = np.zeros(X.shape[1], dtype=float)
    for col in range(X.shape[1]):
        X_perm = X.copy()
        rng.shuffle(X_perm[:, col])
        pred = np.asarray(predict_fn(X_perm)).ravel()
        if task == "binary":
            score = np.mean((pred >= 0.5).astype(int) == y.astype(int))
        else:
            score = -mean_squared_error(y, pred)
        importances[col] = max(0.0, baseline_score - score)
    return importances


def shap_or_permutation_tabular(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
    y_explain: Optional[np.ndarray],
    feature_names: Sequence[str],
    task: str = "regression",
    seed: Optional[int] = None,
    language: str = "pt",
    model_hint: Optional[Any] = None,
) -> ExplainabilityArtifact:
    labels = _labels(language)
    X_background = _safe_sample(np.asarray(X_background, dtype=float), max_rows=30)
    X_explain = _safe_sample(np.asarray(X_explain, dtype=float), max_rows=MAX_EXPLAINABILITY_ROWS)
    names = _safe_feature_names(feature_names, X_explain.shape[1])

    method = "SHAP"
    values: Optional[np.ndarray] = None

    if shap is not None:
        try:
            if model_hint is not None and model_hint.__class__.__name__.lower().startswith("randomforest"):
                explainer = shap.TreeExplainer(model_hint)
                raw = explainer.shap_values(X_explain)
            else:
                explainer = shap.KernelExplainer(lambda data: np.asarray(predict_fn(data)).ravel(), X_background)
                raw = explainer.shap_values(X_explain, nsamples=min(80, 2 * X_explain.shape[1] + 32))

            if isinstance(raw, list):
                raw = raw[-1]
            values = np.asarray(raw)
            if values.ndim == 3:
                values = values[..., -1]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            importances = np.mean(np.abs(values), axis=0)
        except Exception:
            method = labels["fallback"]
            values = None
    else:
        method = labels["fallback"]

    if values is None:
        if y_explain is None:
            importances = np.std(X_explain, axis=0)
        else:
            importances = _permutation_importance(predict_fn, X_explain, y_explain, task, seed)

    image = _bar_chart(labels["feature_importance"], names, importances, labels)
    order = np.argsort(importances)[::-1][:MAX_FEATURES_ON_CHART]
    top = [{"feature": names[i], "importance": float(importances[i])} for i in order]

    return ExplainabilityArtifact(
        key="feature_importance",
        title=f"{labels['feature_importance']} ({method})",
        description=(
            "Mostra quais variáveis mais alteram a saída do modelo. Quando SHAP não está disponível, usa permutação como alternativa segura."
            if not (language or "").lower().startswith("en")
            else "Shows which variables most affect model output. When SHAP is unavailable, permutation importance is used as a safe fallback."
        ),
        image=image,
        summary={"available": True, "method": method, "top_features": top},
        kind="importance",
    )


def classification_confusion_artifact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    language: str = "pt",
) -> ExplainabilityArtifact:
    labels = _labels(language)
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (np.asarray(y_prob).astype(float).ravel() >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    image = _heatmap(labels["confusion_matrix"], cm, ["0", "1"], ["0", "1"], labels)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    return ExplainabilityArtifact(
        key="confusion_matrix",
        title=labels["confusion_matrix"],
        description=(
            "Traduz acertos e erros em contagens: verdadeiro negativo, falso positivo, falso negativo e verdadeiro positivo."
            if not (language or "").lower().startswith("en")
            else "Turns hits and errors into counts: true negatives, false positives, false negatives, and true positives."
        ),
        image=image,
        summary={"available": True, "threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp},
        kind="confusion_matrix",
    )


def regression_confusion_artifact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    language: str = "pt",
    bins: int = 3,
) -> ExplainabilityArtifact:
    labels = _labels(language)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    unique = np.unique(y_true[~np.isnan(y_true)])
    if unique.size < 2:
        return ExplainabilityArtifact(
            key="regression_confusion_matrix",
            title=labels["confusion_matrix"],
            description=labels["not_applicable"],
            summary={"available": False, "reason": "target has fewer than two distinct values"},
            kind="confusion_matrix",
        )

    q = np.linspace(0, 1, min(bins, unique.size) + 1)
    edges = np.unique(np.quantile(y_true, q))
    if edges.size < 3:
        edges = np.linspace(float(np.min(y_true)), float(np.max(y_true)), min(bins, unique.size) + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    true_bins = np.digitize(y_true, edges[1:-1], right=True)
    pred_bins = np.digitize(y_pred, edges[1:-1], right=True)
    n_bins = len(edges) - 1
    cm = confusion_matrix(true_bins, pred_bins, labels=list(range(n_bins)))

    bin_labels = []
    for i in range(n_bins):
        if i == 0:
            bin_labels.append("low" if (language or "").lower().startswith("en") else "baixo")
        elif i == n_bins - 1:
            bin_labels.append("high" if (language or "").lower().startswith("en") else "alto")
        else:
            bin_labels.append("medium" if (language or "").lower().startswith("en") else "médio")

    image = _heatmap(labels["confusion_matrix"], cm, bin_labels, bin_labels, labels)
    return ExplainabilityArtifact(
        key="regression_confusion_matrix",
        title=(
            "Matriz por faixas da regressão" if not (language or "").lower().startswith("en") else "Regression band confusion matrix"
        ),
        description=(
            "Agrupa valores reais e preditos em faixas para mostrar se o modelo erra pouco, superestima ou subestima."
            if not (language or "").lower().startswith("en")
            else "Groups actual and predicted values into bands to show whether the model is close, overestimating, or underestimating."
        ),
        image=image,
        summary={"available": True, "bands": bin_labels, "matrix": cm.tolist()},
        kind="confusion_matrix",
    )



def multiclass_confusion_artifact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    language: str = "pt",
) -> ExplainabilityArtifact:
    labels = _labels(language)
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.argmax(np.asarray(y_prob, dtype=float), axis=1)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    names = [str(class_names[i]) if class_names and i < len(class_names) else str(i) for i in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    image = _heatmap(labels["confusion_matrix"], cm, names, names, labels)
    correct = int(np.trace(cm))
    total = int(np.sum(cm))
    return ExplainabilityArtifact(
        key="confusion_matrix",
        title=labels["confusion_matrix"],
        description=(
            "Mostra, classe por classe, quantas imagens foram classificadas corretamente e onde ocorreram confusões."
            if not (language or "").lower().startswith("en")
            else "Shows, class by class, how many images were classified correctly and where confusions occurred."
        ),
        image=image,
        summary={"available": True, "classes": names, "correct": correct, "total": total, "accuracy_from_matrix": correct / total if total else None},
        kind="confusion_matrix",
    )


def multiclass_reliability_artifact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    language: str = "pt",
    n_bins: int = 8,
) -> ExplainabilityArtifact:
    labels = _labels(language)
    probs = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true).astype(int).ravel()
    if probs.ndim != 2 or probs.shape[0] == 0:
        return ExplainabilityArtifact(
            key="reliability_diagram",
            title=labels["reliability"],
            description=labels["not_applicable"],
            summary={"available": False, "reason": "invalid probabilities"},
            kind="calibration",
        )
    confidence = np.max(probs, axis=1)
    predicted = np.argmax(probs, axis=1)
    correctness = (predicted == y_true).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf, acc = [], []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidence >= left) & (confidence <= right if right == 1 else confidence < right)
        if np.any(mask):
            mean_conf.append(float(np.mean(confidence[mask])))
            acc.append(float(np.mean(correctness[mask])))
    if not mean_conf:
        mean_conf, acc = [float(np.mean(confidence))], [float(np.mean(correctness))]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", label=labels["perfect"])
    ax.plot(mean_conf, acc, marker="o", label=labels["reliability"])
    ax.set_title(labels["reliability"])
    ax.set_xlabel(labels["probability"])
    ax.set_ylabel("Accuracy" if (language or "").lower().startswith("en") else "Acurácia")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ExplainabilityArtifact(
        key="reliability_diagram",
        title=labels["reliability"],
        description=(
            "Compara a confiança média da classe prevista com a acurácia observada em cada faixa de confiança."
            if not (language or "").lower().startswith("en")
            else "Compares the average confidence of the predicted class with observed accuracy in each confidence band."
        ),
        image=_encode_figure(fig),
        summary={"available": True, "task": "multiclass", "mean_confidence": mean_conf, "accuracy": acc},
        kind="calibration",
    )

def reliability_artifact(
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    task: str = "regression",
    language: str = "pt",
    n_bins: int = 8,
) -> ExplainabilityArtifact:
    labels = _labels(language)
    y_true = np.asarray(y_true).ravel()
    y_pred_or_prob = np.asarray(y_pred_or_prob, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    if task == "binary":
        valid = np.isfinite(y_pred_or_prob) & np.isfinite(y_true)
        y_true_i = y_true[valid].astype(int)
        y_prob = np.clip(y_pred_or_prob[valid], 0.0, 1.0)
        if len(np.unique(y_true_i)) < 2:
            plt.close(fig)
            return ExplainabilityArtifact(
                key="reliability_diagram",
                title=labels["reliability"],
                description=labels["not_applicable"],
                summary={"available": False, "reason": "test set has one class only"},
                kind="calibration",
            )
        frac_pos, mean_pred = calibration_curve(y_true_i, y_prob, n_bins=min(n_bins, max(2, len(y_true_i))), strategy="uniform")
        ax.plot([0, 1], [0, 1], linestyle="--", label=labels["perfect"])
        ax.plot(mean_pred, frac_pos, marker="o", label=labels["reliability"])
        ax.set_xlabel(labels["probability"])
        ax.set_ylabel(labels["fraction"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        summary = {
            "available": True,
            "task": "binary",
            "mean_predicted_probability": [float(v) for v in mean_pred],
            "fraction_of_positives": [float(v) for v in frac_pos],
        }
    else:
        valid = np.isfinite(y_pred_or_prob) & np.isfinite(y_true.astype(float))
        yt = y_true.astype(float)[valid]
        yp = y_pred_or_prob[valid]
        if len(yt) < 4:
            plt.close(fig)
            return ExplainabilityArtifact(
                key="reliability_diagram",
                title=labels["reliability"],
                description=labels["not_applicable"],
                summary={"available": False, "reason": "not enough samples"},
                kind="calibration",
            )
        order = np.argsort(yp)
        chunks = np.array_split(order, min(n_bins, max(2, len(order) // 2)))
        mean_pred = np.array([float(np.mean(yp[idx])) for idx in chunks if len(idx)])
        mean_true = np.array([float(np.mean(yt[idx])) for idx in chunks if len(idx)])
        min_axis = float(min(np.min(mean_pred), np.min(mean_true)))
        max_axis = float(max(np.max(mean_pred), np.max(mean_true)))
        ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", label=labels["perfect"])
        ax.plot(mean_pred, mean_true, marker="o", label=labels["reliability"])
        ax.set_xlabel(labels["mean_predicted"])
        ax.set_ylabel(labels["observed"])
        summary = {
            "available": True,
            "task": "regression",
            "mean_predicted": [float(v) for v in mean_pred],
            "mean_observed": [float(v) for v in mean_true],
        }

    ax.set_title(labels["reliability"])
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ExplainabilityArtifact(
        key="reliability_diagram",
        title=labels["reliability"],
        description=(
            "Compara confiança/predição média com resultado observado para avaliar calibração."
            if not (language or "").lower().startswith("en")
            else "Compares confidence/mean prediction with observed outcome to assess calibration."
        ),
        image=_encode_figure(fig),
        summary=summary,
        kind="calibration",
    )


def _find_last_conv1d_layer(model: Any) -> Optional[Any]:
    if tf is None:
        return None
    for layer in reversed(getattr(model, "layers", [])):
        if isinstance(layer, tf.keras.layers.Conv1D):
            return layer
    return None


def gradcam_1d_artifact(
    model: Any,
    example: np.ndarray,
    labels_or_names: Sequence[str],
    task: str = "binary",
    language: str = "pt",
) -> ExplainabilityArtifact:
    labels = _labels(language)
    if tf is None:
        return ExplainabilityArtifact(
            key="gradcam_1d",
            title=labels["gradcam"],
            description=labels["not_applicable"],
            summary={"available": False, "reason": "tensorflow unavailable"},
            kind="gradcam",
        )

    conv_layer = _find_last_conv1d_layer(model)
    if conv_layer is None:
        return ExplainabilityArtifact(
            key="gradcam_1d",
            title=labels["gradcam"],
            description=labels["not_applicable"],
            summary={"available": False, "reason": "no Conv1D layer found"},
            kind="gradcam",
        )

    x = np.asarray(example)
    if x.ndim == 1:
        if np.issubdtype(x.dtype, np.integer):
            model_input = x.reshape(1, -1)
        else:
            model_input = x.reshape(1, -1, 1)
    elif x.ndim == 2:
        model_input = x.reshape(1, *x.shape)
    else:
        model_input = x

    try:
        grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tf.convert_to_tensor(model_input))
            score = predictions[:, 0]
        grads = tape.gradient(score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=1)
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads[0], axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > EPS:
            heatmap = heatmap / np.max(heatmap)
    except Exception:
        return ExplainabilityArtifact(
            key="gradcam_1d",
            title=labels["gradcam"],
            description=labels["not_applicable"],
            summary={"available": False, "reason": "Grad-CAM calculation failed"},
            kind="gradcam",
        )

    n = len(heatmap)
    names = [str(v) for v in labels_or_names]
    if len(names) < n:
        names.extend([str(i + 1) for i in range(len(names), n)])
    names = names[:n]

    fig, ax = plt.subplots(figsize=(max(7.2, n * 0.28), 3.4))
    ax.bar(range(n), heatmap)
    ax.set_title(labels["gradcam"])
    ax.set_xlabel(labels["token_position"])
    ax.set_ylabel(labels["score"])
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)

    order = np.argsort(heatmap)[::-1][:min(MAX_FEATURES_ON_CHART, n)]
    top = [{"item": names[i], "score": float(heatmap[i])} for i in order]
    return ExplainabilityArtifact(
        key="gradcam_1d",
        title=labels["gradcam"],
        description=(
            "Aplica uma versão 1D do Grad-CAM para destacar posições, tokens ou features que mais ativaram a CNN."
            if not (language or "").lower().startswith("en")
            else "Applies a 1D Grad-CAM variant to highlight positions, tokens, or features that most activated the CNN."
        ),
        image=_encode_figure(fig),
        summary={"available": True, "top_activations": top},
        kind="gradcam",
    )


def token_perturbation_artifact(
    model: Any,
    sequence: np.ndarray,
    tokenizer: Any,
    language: str = "pt",
) -> ExplainabilityArtifact:
    labels = _labels(language)
    seq = np.asarray(sequence).astype(int).ravel()
    nonzero_positions = [i for i, token_id in enumerate(seq) if token_id != 0]
    if not nonzero_positions:
        return ExplainabilityArtifact(
            key="token_importance",
            title=labels["feature_importance"],
            description=labels["not_applicable"],
            summary={"available": False, "reason": "empty sequence"},
            kind="importance",
        )

    index_word = getattr(tokenizer, "index_word", {}) or {}
    base = float(model.predict(seq.reshape(1, -1), verbose=0).ravel()[0])
    scores: List[float] = []
    tokens: List[str] = []
    for pos in nonzero_positions[:80]:
        altered = seq.copy()
        altered[pos] = 0
        pred = float(model.predict(altered.reshape(1, -1), verbose=0).ravel()[0])
        scores.append(abs(base - pred))
        tokens.append(str(index_word.get(int(seq[pos]), f"#{int(seq[pos])}")))

    image = _bar_chart(labels["feature_importance"], tokens, scores, labels)
    order = np.argsort(scores)[::-1][:MAX_FEATURES_ON_CHART]
    top = [{"token": tokens[i], "importance": float(scores[i])} for i in order]
    return ExplainabilityArtifact(
        key="token_importance",
        title=labels["feature_importance"],
        description=(
            "Remove cada token individualmente e mede quanto a probabilidade final muda."
            if not (language or "").lower().startswith("en")
            else "Removes each token individually and measures how much the final probability changes."
        ),
        image=image,
        summary={"available": True, "method": "token perturbation", "top_tokens": top},
        kind="importance",
    )



def _to_rgb_images(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images, dtype=float)
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.clip(arr, 0.0, 1.0)


def _image_saliency_fallback(model: Any, image: np.ndarray, class_index: int) -> np.ndarray:
    if tf is None:
        return np.zeros(image.shape[:2], dtype=float)
    x = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x, training=False)
        score = pred[:, class_index]
    grads = tape.gradient(score, x)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    saliency = np.maximum(saliency, 0)
    if np.max(saliency) > EPS:
        saliency = saliency / np.max(saliency)
    return saliency


def shap_image_artifact(
    model: Any,
    X_explain: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    language: str = "pt",
    max_evals: int = 100,
) -> ExplainabilityArtifact:
    """Render the CNN image explanation using IMAT's readable two-panel layout.

    This follows the visual strategy from the reference IMAT 4.2 project: the
    original input image appears on the left and a pixel-level importance map
    appears on the right under the title "SHAP Implementation for Image Data".
    The raw shap.image_plot output is intentionally not used here because, for
    tiny 28x28 MNIST-like images, it can produce colorbar layouts or coarse
    bands that are harder for users to interpret.
    """
    labels = _labels(language)
    if X_explain is None or len(X_explain) == 0:
        return ExplainabilityArtifact(
            key="image_shap",
            title=labels["image_shap"],
            description=labels["not_applicable"],
            summary={"available": False, "reason": "no image available"},
            kind="image_shap",
        )

    image = np.asarray(X_explain[:1], dtype="float32")
    if image.ndim == 3:
        image = image[..., np.newaxis]

    pred = np.asarray(model.predict(image, verbose=0))[0]
    class_index = int(np.argmax(pred))
    class_label = str(class_names[class_index]) if class_names and class_index < len(class_names) else str(class_index)

    saliency = _image_saliency_fallback(model, image[0], class_index)
    display = _to_rgb_images(image)[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    axes[0].imshow(display.squeeze() if display.shape[-1] == 1 else display, cmap="gray")
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(display.squeeze() if display.shape[-1] == 1 else display, cmap="gray")
    axes[1].imshow(saliency, alpha=0.55)
    axes[1].set_title(labels["image_shap"])
    axes[1].axis("off")

    return ExplainabilityArtifact(
        key="image_shap",
        title=labels["image_shap"],
        description=(
            "Mostra a imagem de entrada e, ao lado, um mapa visual de importância dos pixels para apoiar a interpretação da classe prevista."
            if not (language or "").lower().startswith("en")
            else "Shows the input image and, next to it, a visual pixel-importance map to support interpretation of the predicted class."
        ),
        image=_encode_figure(fig),
        summary={
            "available": True,
            "method": "IMAT image saliency layout",
            "predicted_class": class_label,
            "predicted_probability": float(pred[class_index]),
        },
        kind="image_shap",
    )

def artifacts_to_payload(artifacts: Iterable[ExplainabilityArtifact]) -> Dict[str, Any]:
    items = [artifact.to_dict() for artifact in artifacts]
    return {"items": items}


def artifacts_text_summary(artifacts: Iterable[ExplainabilityArtifact], language: str = "pt") -> str:
    lines: List[str] = []
    is_en = (language or "").lower().startswith("en")
    header = "EXPLAINABILITY DIAGNOSTICS" if is_en else "DIAGNÓSTICOS DE EXPLICABILIDADE"
    lines.append(header)
    for artifact in artifacts:
        data = artifact.to_dict()
        summary = data.get("summary") or {}
        lines.append(f"- {data.get('title')}: {data.get('description')}")
        if summary:
            compact = str(summary)
            if len(compact) > 900:
                compact = compact[:900] + "..."
            lines.append(f"  summary={compact}")
    return "\n".join(lines)

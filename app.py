from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import re
import tempfile
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback

from model_analysis_module.utils import set_seed
from model_analysis_module.models import train_mlp
from model_analysis_module.metrics import (
    compute_metrics,
    compute_classification_metrics
)
from model_analysis_module.flowchart import create_flowchart
from model_analysis_module.explanation import (
    compose_prompt,
    compose_short_prompt,
    get_openrouter_explanation,
    get_short_explanation,
)

from model_analysis_module.cnn import (
    prepare_text_data,
    build_cnn_text_model,
    create_cnn_flowchart_full,
    create_cnn_flowchart_summary,
    build_cnn_numeric_model,
    create_cnn_flowchart_full_numeric,
    create_cnn_flowchart_summary_numeric,
    load_image_npz,
    build_cnn_image_model,
    create_cnn_image_flowchart
)

from model_analysis_module.random_forest import (
    train_rf_reg,
    create_rf_flowchart_full_reg,
    create_rf_flowchart_summary_reg
)

from model_analysis_module.explainability import (
    artifacts_text_summary,
    artifacts_to_payload,
    classification_confusion_artifact,
    gradcam_1d_artifact,
    history_loss_curve,
    random_forest_tree_curve,
    regression_confusion_artifact,
    reliability_artifact,
    multiclass_confusion_artifact,
    multiclass_reliability_artifact,
    shap_image_artifact,
    shap_or_permutation_tabular,
    token_perturbation_artifact,
)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

CORS(app)


# ========================= páginas / saúde =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ========================= helpers =========================

_DECIMAL_WITH_MORE_THAN_TWO_PLACES = re.compile(r"(?<![\w])-?\d+\.\d{3,}(?!\d)")
_DECIMAL_COMMA_WITH_MORE_THAN_TWO_PLACES = re.compile(r"(?<![\w])-?\d+,\d{3,}(?!\d)")
_PROGRESS_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
_PROGRESS_DIR = os.path.join(tempfile.gettempdir(), "imat-training-progress")
_PROGRESS_MAX_AGE_SECONDS = 2 * 60 * 60


def _valid_progress_id(progress_id):
    return bool(progress_id and _PROGRESS_ID_PATTERN.fullmatch(progress_id))


def _progress_file_path(progress_id):
    return os.path.join(_PROGRESS_DIR, f"{progress_id}.json")


def _cleanup_stale_progress_files():
    """Removes old progress snapshots without affecting an active analysis."""
    try:
        os.makedirs(_PROGRESS_DIR, exist_ok=True)
        cutoff = time.time() - _PROGRESS_MAX_AGE_SECONDS
        for filename in os.listdir(_PROGRESS_DIR):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(_PROGRESS_DIR, filename)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                continue
    except OSError:
        # Progress reporting must never prevent a model run from completing.
        pass


class _TrainingProgress:
    """Persists real workflow milestones so the browser can poll them."""

    def __init__(self, progress_id):
        self.progress_id = progress_id if _valid_progress_id(progress_id) else None
        self.percent = 0

    @classmethod
    def from_request(cls):
        progress = cls(request.args.get("progress_id"))
        if progress.enabled:
            _cleanup_stale_progress_files()
        return progress

    @property
    def enabled(self):
        return self.progress_id is not None

    def update(
        self,
        percent,
        stage,
        *,
        status="running",
        current=None,
        total=None,
        unit=None,
    ):
        if not self.enabled:
            return

        bounded_percent = max(self.percent, min(100, int(round(percent))))
        self.percent = bounded_percent
        payload = {
            "status": status,
            "stage": stage,
            "percent": bounded_percent,
            "updated_at": time.time(),
        }
        if current is not None:
            payload["current"] = int(current)
        if total is not None:
            payload["total"] = int(total)
        if unit:
            payload["unit"] = unit

        try:
            os.makedirs(_PROGRESS_DIR, exist_ok=True)
            path = _progress_file_path(self.progress_id)
            temp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
            try:
                with open(temp_path, "w", encoding="utf-8") as progress_file:
                    json.dump(payload, progress_file, ensure_ascii=False)
                os.replace(temp_path, path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except OSError:
            pass

    def update_training_step(self, current, total, *, unit):
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        percent = 18 + ((65 - 18) * current / total)
        stage = "training_epochs" if unit == "epoch" else "training_trees"
        self.update(
            percent,
            stage,
            current=current,
            total=total,
            unit=unit,
        )

    def complete(self):
        self.update(100, "completed", status="completed")

    def fail(self):
        if self.enabled:
            try:
                with open(_progress_file_path(self.progress_id), "r", encoding="utf-8") as progress_file:
                    existing = json.load(progress_file)
                self.percent = max(self.percent, int(existing.get("percent", 0)))
            except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
        self.update(self.percent, "failed", status="failed")


class _EpochProgressCallback(Callback):
    """Reports epochs only after Keras confirms that each one has ended."""

    def __init__(self, progress, total_epochs):
        super().__init__()
        self.progress = progress
        self.total_epochs = max(1, int(total_epochs))

    def on_train_begin(self, logs=None):
        self.progress.update_training_step(0, self.total_epochs, unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        self.progress.update_training_step(epoch + 1, self.total_epochs, unit="epoch")


def _run_progress_steps(progress, builders, *, stage, start_percent, end_percent):
    """Runs concrete analysis steps and advances only after each one finishes."""
    builders = list(builders)
    total = len(builders)
    progress.update(start_percent, stage, current=0, total=total)
    results = []

    for index, builder in enumerate(builders, start=1):
        results.append(builder())
        percent = start_percent + ((end_percent - start_percent) * index / max(1, total))
        progress.update(percent, stage, current=index, total=total)

    return results


@app.route("/training-progress/<progress_id>", methods=["GET"])
def training_progress(progress_id):
    if not _valid_progress_id(progress_id):
        return jsonify({"error": "Invalid progress identifier"}), 400

    try:
        with open(_progress_file_path(progress_id), "r", encoding="utf-8") as progress_file:
            payload = json.load(progress_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = {
            "status": "pending",
            "stage": "starting",
            "percent": 0,
            "updated_at": time.time(),
        }

    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response, 200


@app.after_request
def _mark_failed_progress(response):
    progress_id = request.args.get("progress_id")
    if response.status_code >= 400 and _valid_progress_id(progress_id):
        _TrainingProgress(progress_id).fail()
    return response


def _force_two_decimal_numbers(text: str) -> str:
    """Keeps numeric evidence sent to the LLM with at most two decimal places."""
    def _replace_dot(match):
        try:
            return f"{float(match.group(0)):.2f}"
        except Exception:
            return match.group(0)

    def _replace_comma(match):
        try:
            return f"{float(match.group(0).replace(',', '.')):.2f}".replace('.', ',')
        except Exception:
            return match.group(0)

    value = _DECIMAL_WITH_MORE_THAN_TWO_PLACES.sub(_replace_dot, str(text or ""))
    return _DECIMAL_COMMA_WITH_MORE_THAN_TWO_PLACES.sub(_replace_comma, value)

def _fmt_secs(v):
    try:
        v = float(v)
        return f"{v:.2f}s"
    except Exception:
        return "—"


def _fmt_oob(v):
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def _compute_rf_stats_fallback(rf_model):
    """Gera estatísticas básicas quando train_rf_reg não as retorna."""
    depths, nodes = [], []

    for est in getattr(rf_model, "estimators_", []):
        t = est.tree_
        depths.append(t.max_depth)
        nodes.append(t.node_count)

    avg_depth = float(np.mean(depths)) if depths else float("nan")
    avg_nodes = float(np.mean(nodes)) if nodes else float("nan")
    oob = getattr(rf_model, "oob_score_", None)

    return {
        "fit_time": float("nan"),
        "oob_score": oob,
        "avg_depth": avg_depth,
        "avg_nodes": avg_nodes,
    }


def _get_ui_language():
    value = (
        request.form.get("ui_language")
        or request.form.get("language")
        or request.args.get("ui_language")
        or request.args.get("language")
        or request.headers.get("X-UI-Language")
        or request.headers.get("X-Language")
        or request.headers.get("Accept-Language")
        or "pt"
    )
    value = value.strip().lower()
    return "en" if value.startswith("en") else "pt"


def _t(ui_language, pt_text, en_text):
    return en_text if ui_language == "en" else pt_text



def _build_explanation_payload(
    style,
    objective,
    ctx_frontend,
    ctx_train,
    style_free,
    ui_language,
    progress=None,
):
    """Generates both explanation modes returned to the frontend.

    long: HTML narrative that may receive inline explainability charts in the UI.
    short: plain text, 1000-1500 chars, with no charts or visual references.
    """
    ctx_frontend = _force_two_decimal_numbers(ctx_frontend)
    ctx_train = _force_two_decimal_numbers(ctx_train)

    long_prompt = compose_prompt(
        style=style,
        objective=objective,
        ctx_frontend=ctx_frontend,
        ctx_train=ctx_train,
        style_custom=style_free,
        language=ui_language,
    )
    short_prompt = compose_short_prompt(
        style=style,
        objective=objective,
        ctx_frontend=ctx_frontend,
        ctx_train=ctx_train,
        style_custom=style_free,
        language=ui_language,
    )

    if progress:
        progress.update(88, "generating_detailed_explanation")
    long_explanation = get_openrouter_explanation(long_prompt, language=ui_language)

    if progress:
        progress.update(94, "generating_short_explanation")
    short_explanation = get_short_explanation(short_prompt, language=ui_language, style=style)

    if progress:
        progress.update(98, "finalizing")

    return {
        "long": long_explanation,
        "short": short_explanation,
    }


# ========================= MLP (regressão) =========================
@app.route("/upload", methods=["POST"])
def upload_mlp():
    progress = _TrainingProgress.from_request()
    progress.update(2, "reading_data")

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file).dropna()
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {e}"}), 400

    numeric = df.select_dtypes(include="number")
    target = request.form.get("target_variable")

    if not target or target not in numeric.columns:
        return jsonify({"error": "Invalid target variable"}), 400

    try:
        progress.update(10, "preparing_data")
        ui_language = _get_ui_language()

        feature_names = list(numeric.drop(columns=[target]).columns)
        X = numeric.drop(columns=[target]).values.astype(float)
        y = numeric[target].values.astype(float)

        layers = list(map(int, request.form["layers"].split(",")))
        activation = request.form["activation"]
        optimizer = request.form["optimizer"]
        loss = request.form["loss"]
        epochs = int(request.form["epochs"])
        batch_size = int(request.form["batch_size"])

        seed_param = request.form.get("seed")
        seed = int(seed_param) if seed_param else None

        style = request.form.get("ex_prompt", "exec_summary")
        objective = request.form.get("objective", "")
        style_free = request.form.get("ex_prompt_custom", "")

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        progress.update(15, "building_model")
        model = train_mlp(
            input_shape=X.shape[1],
            layers=layers,
            activation=activation,
            optimizer=optimizer,
            loss=loss,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed
        )

        t0 = time.time()
        history = model.fit(
            Xtr,
            ytr,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[_EpochProgressCallback(progress, epochs)],
        )
        fit_time = time.time() - t0

        progress.update(67, "evaluating")
        y_pred_test = model.predict(Xte, verbose=0).ravel()
        metrics = compute_metrics(yte, y_pred_test)

        mlp_predict = lambda data: model.predict(np.asarray(data, dtype=float), verbose=0).ravel()
        explainability_artifacts = _run_progress_steps(
            progress,
            [
                lambda: history_loss_curve(history, language=ui_language),
                lambda: shap_or_permutation_tabular(
                    predict_fn=mlp_predict,
                    X_background=Xtr,
                    X_explain=Xte,
                    y_explain=yte,
                    feature_names=feature_names,
                    task="regression",
                    seed=seed,
                    language=ui_language,
                ),
                lambda: regression_confusion_artifact(yte, y_pred_test, language=ui_language),
                lambda: reliability_artifact(yte, y_pred_test, task="regression", language=ui_language),
            ],
            stage="building_diagnostics",
            start_percent=70,
            end_percent=84,
        )

        pred0 = model.predict(Xte[0:1], verbose=0)
        Ws, Bs = zip(*(layer.get_weights() for layer in model.layers))

        progress.update(84, "building_visualizations", current=0, total=2)
        b64_d, info = create_flowchart(
            model,
            pred0,
            Ws,
            Bs,
            mse=metrics["mse"],
            max_display_neurons=5,
            return_info=True
        )
        progress.update(86, "building_visualizations", current=1, total=2)

        b64_s = create_flowchart(
            model,
            pred0,
            Ws,
            Bs,
            mse=metrics["mse"],
            max_display_neurons=5,
            return_info=False
        )
        progress.update(87, "building_visualizations", current=2, total=2)

        hist = history.history
        key = "val_loss" if "val_loss" in hist else "loss"
        best_epoch = int(np.argmin(hist[key]))
        best_stats = {k: float(hist[k][best_epoch]) for k in hist}
        last_epoch_stats = {k: float(v[-1]) for k, v in hist.items()}

        ctx_frontend = (
            f"{_t(ui_language, 'Modelo', 'Model')}: {_t(ui_language, 'MLP (regressão)', 'MLP (regression)')}\n"
            f"layers={layers}, activation={activation}, optimizer={optimizer}, loss={loss}, "
            f"epochs={epochs}, batch_size={batch_size}, seed={seed}\n"
            f"Target: {target}"
        )

        ctx_train = (
            f"{_t(ui_language, 'Dataset', 'Dataset')}: {_t(ui_language, 'n_amostras', 'n_samples')}={len(X)}, n_features={X.shape[1]}\n"
            f"Split: train={len(Xtr)}, test={len(Xte)}, {_t(ui_language, 'val=20% do treino', 'val=20% of training')}\n"
            f"{_t(ui_language, 'Tempo de treino', 'Training time')}: {_fmt_secs(fit_time)} | {_t(ui_language, 'Melhor época', 'Best epoch')} ({key}): {best_epoch}\n"
            f"{_t(ui_language, 'Treino (última época)', 'Training (last epoch)')}: {last_epoch_stats}\n"
            f"{_t(ui_language, 'Treino (melhor época)', 'Training (best epoch)')}: {best_stats}\n"
            f"{_t(ui_language, 'Métricas de TESTE', 'TEST metrics')}: {metrics}\n"
            f"{_t(ui_language, 'Resumo do grafo (amostra)', 'Graph summary (sample)')}:\n" + "\n".join(info[:20]) +
            "\n\n" + artifacts_text_summary(explainability_artifacts, language=ui_language)
        )

        explanations = _build_explanation_payload(
            style=style,
            objective=objective,
            ctx_frontend=ctx_frontend,
            ctx_train=ctx_train,
            style_free=style_free,
            ui_language=ui_language,
            progress=progress,
        )

        progress.complete()
        return jsonify({
            "detailed": b64_d,
            "summarized": b64_s,
            "explanation": explanations["long"],
            "short_explanation": explanations["short"],
            "explanations": explanations,
            "explainability": artifacts_to_payload(explainability_artifacts),
            "ui_language_used": ui_language,
            **metrics
        }), 200

    except Exception as e:
        progress.fail()
        return jsonify({"error": f"MLP processing failed: {e}"}), 500


# ========================= Random Forest (regressão) =========================
@app.route("/upload_rf", methods=["POST"])
def upload_rf():
    progress = _TrainingProgress.from_request()
    progress.update(2, "reading_data")

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file).dropna()
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {e}"}), 400

    target = request.form.get("target_variable")
    if not target or target not in df.select_dtypes(include="number").columns:
        return jsonify({"error": "Invalid target variable"}), 400

    try:
        progress.update(10, "preparing_data")
        ui_language = _get_ui_language()

        n_estimators = int(request.form["n_estimators"])

        max_depth = request.form.get("max_depth")
        max_depth = int(max_depth) if max_depth else None

        seed_param = request.form.get("seed")
        seed = int(seed_param) if seed_param else None

        progress.update(15, "building_model")
        rf_model, Xtrain, Xtest, ytrain, ytest, rf_stats = train_rf_reg(
            df,
            target,
            n_estimators=n_estimators,
            max_depth=max_depth,
            seed=seed,
            progress_callback=(
                (
                    lambda current, total: progress.update_training_step(
                        current,
                        total,
                        unit="tree",
                    )
                )
                if progress.enabled
                else None
            ),
        )

        progress.update(67, "evaluating")
        y_pred = rf_model.predict(Xtest)
        metrics = compute_metrics(ytest, y_pred)

        example_vec = Xtest[0]
        feature_names = list(df.select_dtypes(include="number").columns.drop(target))

        progress.update(70, "building_visualizations", current=0, total=2)
        b64_d = create_rf_flowchart_full_reg(
            rf_model,
            example_vec,
            feature_names=feature_names,
            dpi=1999
        )
        progress.update(72, "building_visualizations", current=1, total=2)

        b64_s = create_rf_flowchart_summary_reg(
            rf_model,
            example_vec,
            feature_names=feature_names
        )
        progress.update(74, "building_visualizations", current=2, total=2)

        style = request.form.get("ex_prompt", "exec_summary")
        objective = request.form.get("objective", "")
        style_free = request.form.get("ex_prompt_custom", "")

        importances = getattr(rf_model, "feature_importances_", None)
        top_feats = []
        if importances is not None:
            order = np.argsort(importances)[::-1][:10]
            top_feats = [(feature_names[i], float(importances[i])) for i in order]

        explainability_artifacts = _run_progress_steps(
            progress,
            [
                lambda: random_forest_tree_curve(
                    rf_model,
                    Xtrain,
                    ytrain,
                    Xtest,
                    ytest,
                    language=ui_language,
                ),
                lambda: shap_or_permutation_tabular(
                    predict_fn=rf_model.predict,
                    X_background=Xtrain,
                    X_explain=Xtest,
                    y_explain=ytest,
                    feature_names=feature_names,
                    task="regression",
                    seed=seed,
                    language=ui_language,
                    model_hint=rf_model,
                ),
                lambda: regression_confusion_artifact(ytest, y_pred, language=ui_language),
                lambda: reliability_artifact(ytest, y_pred, task="regression", language=ui_language),
            ],
            stage="building_diagnostics",
            start_percent=74,
            end_percent=87,
        )

        ft_str = _fmt_secs(rf_stats.get("fit_time"))
        oob_str = _fmt_oob(rf_stats.get("oob_score"))

        ctx_frontend = (
            f"{_t(ui_language, 'Modelo', 'Model')}: {_t(ui_language, 'Random Forest (regressão)', 'Random Forest (regression)')}\n"
            f"n_estimators={n_estimators}, max_depth={max_depth}, seed={seed}\n"
            f"Target: {target}"
        )

        ctx_train = (
            f"Split: train={len(Xtrain)}, test={len(Xtest)}\n"
            f"{_t(ui_language, 'Tempo de treino', 'Training time')}: {ft_str} | OOB: {oob_str}\n"
            f"{_t(ui_language, 'Árvores', 'Trees')}: avg_depth={rf_stats.get('avg_depth')}, avg_nodes={rf_stats.get('avg_nodes')}\n"
            f"{_t(ui_language, 'Top importâncias (10)', 'Top importances (10)')}: {top_feats}\n"
            f"{_t(ui_language, 'Métricas de TESTE', 'TEST metrics')}: {metrics}" +
            "\n\n" + artifacts_text_summary(explainability_artifacts, language=ui_language)
        )

        explanations = _build_explanation_payload(
            style=style,
            objective=objective,
            ctx_frontend=ctx_frontend,
            ctx_train=ctx_train,
            style_free=style_free,
            ui_language=ui_language,
            progress=progress,
        )

        progress.complete()
        return jsonify({
            "detailed": b64_d,
            "summarized": b64_s,
            "explanation": explanations["long"],
            "short_explanation": explanations["short"],
            "explanations": explanations,
            "explainability": artifacts_to_payload(explainability_artifacts),
            "ui_language_used": ui_language,
            **metrics
        }), 200

    except Exception as e:
        progress.fail()
        return jsonify({"error": f"Random Forest processing failed: {e}"}), 500


# ========================= CNN =========================
@app.route("/upload_cnn", methods=["POST"])
def upload_cnn():
    progress = _TrainingProgress.from_request()
    progress.update(2, "reading_data")

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    input_type = (request.form.get("input_type") or "text").lower()
    df = None

    if input_type != "image":
        try:
            df = pd.read_csv(file).dropna(how="all")
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV: {e}"}), 400

    try:
        progress.update(10, "preparing_data")
        ui_language = _get_ui_language()

        embed_dim = int(request.form.get("embed_dim", 50))
        filters = int(request.form.get("filters", 32))
        kernel_size = int(request.form.get("kernel_size", 3))
        dense_units = int(request.form.get("dense_units", 10))
        epochs = int(request.form.get("epochs", 1))
        batch_size = int(request.form.get("batch_size", 16))
        threshold = float(request.form.get("threshold", 0.5))

        seed_param = request.form.get("seed")
        seed = int(seed_param) if seed_param else None
    except Exception as e:
        return jsonify({"error": f"Invalid hyperparameter: {e}"}), 400

    style = request.form.get("ex_prompt", "exec_summary")
    objective = request.form.get("objective", "")
    style_free = request.form.get("ex_prompt_custom", "")

    try:
        if input_type == "image":
            try:
                Xtr, Xte, ytr, yte, class_names = load_image_npz(file, seed=seed)
            except Exception as e:
                return jsonify({"error": f"Failed to read image dataset (.npz): {e}"}), 400

            set_seed(seed)
            num_classes = int(len(np.unique(np.concatenate([ytr, yte]))))
            if not class_names:
                class_names = [str(i) for i in range(num_classes)]

            # Mantém o teste rápido para uso didático da IMAT. O arquivo default já vem reduzido.
            max_train = int(request.form.get("max_train_samples", 1200))
            max_test = int(request.form.get("max_test_samples", 300))
            if len(Xtr) > max_train:
                Xtr = Xtr[:max_train]
                ytr = ytr[:max_train]
            if len(Xte) > max_test:
                Xte = Xte[:max_test]
                yte = yte[:max_test]

            progress.update(15, "building_model")
            model = build_cnn_image_model(
                input_shape=Xtr.shape[1:],
                num_classes=num_classes,
                filters=filters,
                kernel_size=kernel_size,
                dense_units=dense_units,
            )

            t0 = time.time()
            history = model.fit(
                Xtr,
                ytr,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[_EpochProgressCallback(progress, epochs)],
            )
            fit_time = time.time() - t0

            progress.update(67, "evaluating")
            y_prob = model.predict(Xte, verbose=0)
            y_pred = np.argmax(y_prob, axis=1)
            accuracy = float(np.mean(y_pred == yte))
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics_out = {
                "accuracy": accuracy,
                "precision": float(precision_score(yte, y_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(yte, y_pred, average="macro", zero_division=0)),
                "f1": float(f1_score(yte, y_pred, average="macro", zero_division=0)),
            }

            progress.update(70, "building_visualizations", current=0, total=2)
            b64_d = create_cnn_image_flowchart(model, Xtr.shape[1:], num_classes, dpi=1999, detailed=True)
            progress.update(72, "building_visualizations", current=1, total=2)
            b64_s = create_cnn_image_flowchart(model, Xtr.shape[1:], num_classes, dpi=1999, detailed=False)
            progress.update(74, "building_visualizations", current=2, total=2)

            explainability_artifacts = _run_progress_steps(
                progress,
                [
                    lambda: history_loss_curve(history, language=ui_language),
                    lambda: multiclass_confusion_artifact(
                        yte,
                        y_prob,
                        class_names=class_names,
                        language=ui_language,
                    ),
                    lambda: multiclass_reliability_artifact(yte, y_prob, language=ui_language),
                    lambda: shap_image_artifact(
                        model,
                        Xte[:1],
                        class_names=class_names,
                        language=ui_language,
                    ),
                ],
                stage="building_diagnostics",
                start_percent=74,
                end_percent=87,
            )

            hist = history.history
            if "val_loss" in hist:
                key = "val_loss"
                idx = int(np.argmin(hist["val_loss"]))
            elif "val_accuracy" in hist:
                key = "val_accuracy"
                idx = int(np.argmax(hist["val_accuracy"]))
            else:
                key = "loss"
                idx = int(np.argmin(hist["loss"]))

            best_epoch = idx
            best_stats = {k: float(hist[k][best_epoch]) for k in hist}
            last_epoch_stats = {k: float(v[-1]) for k, v in hist.items()}
            balance = dict(zip(*np.unique(ytr, return_counts=True)))

            ctx_frontend = (
                f"{_t(ui_language, 'Modelo', 'Model')}: {_t(ui_language, 'CNN (imagem, classificação multiclasse)', 'CNN (image, multiclass classification)')}\n"
                f"input_shape={Xtr.shape[1:]}, classes={class_names}, filters={filters}, kernel={kernel_size}, "
                f"dense_units={dense_units}, epochs={epochs}, batch_size={batch_size}, seed={seed}\n"
                f"Dataset file: {file.filename}"
            )

            ctx_train = (
                f"Split: train={len(Xtr)}, test={len(Xte)}, {_t(ui_language, 'val=20% do treino', 'val=20% of training')} | balance_train={balance}\n"
                f"{_t(ui_language, 'Tempo de treino', 'Training time')}: {_fmt_secs(fit_time)} | {_t(ui_language, 'Melhor época', 'Best epoch')} ({key}): {best_epoch}\n"
                f"{_t(ui_language, 'Treino (última época)', 'Training (last epoch)')}: {last_epoch_stats}\n"
                f"{_t(ui_language, 'Treino (melhor época)', 'Training (best epoch)')}: {best_stats}\n"
                f"{_t(ui_language, 'Métricas de TESTE', 'TEST metrics')}: {metrics_out}" +
                "\n\n" + artifacts_text_summary(explainability_artifacts, language=ui_language)
            )

            explanations = _build_explanation_payload(
                style=style,
                objective=objective,
                ctx_frontend=ctx_frontend,
                ctx_train=ctx_train,
                style_free=style_free,
                ui_language=ui_language,
                progress=progress,
            )

            progress.complete()
            return jsonify({
                "detailed": b64_d,
                "summarized": b64_s,
                "explanation": explanations["long"],
                "short_explanation": explanations["short"],
                "explanations": explanations,
                "explainability": artifacts_to_payload(explainability_artifacts),
                "ui_language_used": ui_language,
                **metrics_out,
            }), 200

        if input_type == "text":
            text_col = request.form.get("text_column")
            target_col = request.form.get("target_variable")

            if text_col not in df.columns or target_col not in df.columns:
                return jsonify({"error": "Invalid text/target columns"}), 400

            vocab_size = int(request.form.get("vocab_size", 5000))
            maxlen = int(request.form.get("maxlen", 100))

            tok, Xtr, Xte, ytr, yte, raw_te = prepare_text_data(
                df,
                text_col,
                target_col,
                vocab_size,
                maxlen,
                seed=seed
            )

            set_seed(seed)
            progress.update(15, "building_model")
            model = build_cnn_text_model(
                vocab_size,
                embed_dim,
                filters,
                kernel_size,
                dense_units
            )

            t0 = time.time()
            history = model.fit(
                Xtr,
                ytr,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[_EpochProgressCallback(progress, epochs)],
            )
            fit_time = time.time() - t0

            progress.update(67, "evaluating")
            y_prob = model.predict(Xte, verbose=0).ravel()
            metrics_out = compute_classification_metrics(
                yte,
                y_prob,
                threshold=threshold
            )

            example_seq = Xte[0]
            example_text = raw_te[0]

            progress.update(70, "building_visualizations", current=0, total=2)
            b64_d = create_cnn_flowchart_full(
                model,
                example_seq,
                example_text,
                threshold=threshold,
                max_filters=1000,
                dpi=1999
            )
            progress.update(72, "building_visualizations", current=1, total=2)

            b64_s = create_cnn_flowchart_summary(
                model,
                example_seq,
                example_text,
                threshold=threshold,
                dpi=1999
            )
            progress.update(74, "building_visualizations", current=2, total=2)

            example_tokens = [
                str(tok.index_word.get(int(token_id), "PAD" if int(token_id) == 0 else f"#{int(token_id)}"))
                for token_id in example_seq
            ]
            explainability_artifacts = _run_progress_steps(
                progress,
                [
                    lambda: history_loss_curve(history, language=ui_language),
                    lambda: classification_confusion_artifact(
                        yte,
                        y_prob,
                        threshold=threshold,
                        language=ui_language,
                    ),
                    lambda: reliability_artifact(
                        yte,
                        y_prob,
                        task="binary",
                        language=ui_language,
                    ),
                    lambda: gradcam_1d_artifact(
                        model,
                        example_seq,
                        example_tokens,
                        task="binary",
                        language=ui_language,
                    ),
                    lambda: token_perturbation_artifact(
                        model,
                        example_seq,
                        tok,
                        language=ui_language,
                    ),
                ],
                stage="building_diagnostics",
                start_percent=74,
                end_percent=87,
            )

            hist = history.history
            if "val_loss" in hist:
                key = "val_loss"
                idx = int(np.argmin(hist["val_loss"]))
            elif "val_accuracy" in hist:
                key = "val_accuracy"
                idx = int(np.argmax(hist["val_accuracy"]))
            else:
                key = "loss"
                idx = int(np.argmin(hist["loss"]))

            best_epoch = idx
            best_stats = {k: float(hist[k][best_epoch]) for k in hist}
            last_epoch_stats = {k: float(v[-1]) for k, v in hist.items()}

            balance = dict(zip(*np.unique(ytr, return_counts=True)))

            ctx_frontend = (
                f"{_t(ui_language, 'Modelo', 'Model')}: {_t(ui_language, 'CNN (texto, classificação binária)', 'CNN (text, binary classification)')}\n"
                f"vocab={vocab_size}, maxlen={maxlen}, embed_dim={embed_dim}, "
                f"filters={filters}, kernel={kernel_size}, dense_units={dense_units}, "
                f"epochs={epochs}, batch_size={batch_size}, seed={seed}, threshold={threshold}\n"
                f"{_t(ui_language, 'Colunas', 'Columns')}: text={text_col}, target={target_col}"
            )

            ctx_train = (
                f"Split: train={len(Xtr)}, test={len(Xte)}, {_t(ui_language, 'val=20% do treino', 'val=20% of training')} | balance_train={balance}\n"
                f"{_t(ui_language, 'Tempo de treino', 'Training time')}: {_fmt_secs(fit_time)} | {_t(ui_language, 'Melhor época', 'Best epoch')} ({key}): {best_epoch}\n"
                f"{_t(ui_language, 'Treino (última época)', 'Training (last epoch)')}: {last_epoch_stats}\n"
                f"{_t(ui_language, 'Treino (melhor época)', 'Training (best epoch)')}: {best_stats}\n"
                f"{_t(ui_language, 'Métricas de TESTE', 'TEST metrics')}: {metrics_out}\n"
                f"{_t(ui_language, 'Exemplo de texto (teste)', 'Test text example')}: {str(example_text)[:200]}" +
                "\n\n" + artifacts_text_summary(explainability_artifacts, language=ui_language)
            )

            explanations = _build_explanation_payload(
                style=style,
                objective=objective,
                ctx_frontend=ctx_frontend,
                ctx_train=ctx_train,
                style_free=style_free,
                ui_language=ui_language,
                progress=progress,
            )

            progress.complete()
            return jsonify({
                "detailed": b64_d,
                "summarized": b64_s,
                "explanation": explanations["long"],
                "short_explanation": explanations["short"],
                "explanations": explanations,
                "explainability": artifacts_to_payload(explainability_artifacts),
                "ui_language_used": ui_language,
                **metrics_out
            }), 200

        elif input_type == "numeric":
            target_col = request.form.get("target_variable")
            if target_col not in df.columns:
                return jsonify({"error": "Invalid target column"}), 400

            numeric = df.select_dtypes(include="number")
            if target_col not in numeric.columns:
                return jsonify({"error": "Target must be numeric or binary"}), 400

            features = numeric.drop(columns=[target_col]).values.astype(float)
            y = numeric[target_col].values

            task = (request.form.get("task") or "regression").lower()
            if task not in ("regression", "binary"):
                task = "regression"

            X = features.reshape(features.shape[0], features.shape[1], 1)

            Xtr, Xte, ytr, yte = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=seed
            )

            set_seed(seed)
            progress.update(15, "building_model")
            model = build_cnn_numeric_model(
                input_len=X.shape[1],
                filters=filters,
                kernel_size=kernel_size,
                dense_units=dense_units,
                task=task
            )

            t0 = time.time()
            history = model.fit(
                Xtr,
                ytr,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[_EpochProgressCallback(progress, epochs)],
            )
            fit_time = time.time() - t0

            progress.update(67, "evaluating")
            y_pred = model.predict(Xte, verbose=0).ravel()

            metrics_out = (
                compute_classification_metrics(yte, y_pred, threshold=threshold)
                if task == "binary"
                else compute_metrics(yte, y_pred)
            )

            feature_names = list(numeric.drop(columns=[target_col]).columns)
            example_vec = Xte[0].reshape(-1)

            progress.update(70, "building_visualizations", current=0, total=2)
            b64_d = create_cnn_flowchart_full_numeric(
                model,
                example_vec,
                feature_names,
                task=task,
                threshold=threshold,
                dpi=1999
            )
            progress.update(72, "building_visualizations", current=1, total=2)

            b64_s = create_cnn_flowchart_summary_numeric(
                model,
                example_vec,
                feature_names,
                task=task,
                threshold=threshold,
                dpi=1999
            )
            progress.update(74, "building_visualizations", current=2, total=2)

            Xtr_flat = Xtr.reshape(Xtr.shape[0], Xtr.shape[1])
            Xte_flat = Xte.reshape(Xte.shape[0], Xte.shape[1])

            def cnn_numeric_predict(data):
                arr = np.asarray(data, dtype=float)
                return model.predict(
                    arr.reshape(arr.shape[0], arr.shape[1], 1),
                    verbose=0
                ).ravel()

            explainability_artifacts = _run_progress_steps(
                progress,
                [
                    lambda: history_loss_curve(history, language=ui_language),
                    lambda: shap_or_permutation_tabular(
                        predict_fn=cnn_numeric_predict,
                        X_background=Xtr_flat,
                        X_explain=Xte_flat,
                        y_explain=yte,
                        feature_names=feature_names,
                        task=task,
                        seed=seed,
                        language=ui_language,
                    ),
                    lambda: (
                        classification_confusion_artifact(
                            yte,
                            y_pred,
                            threshold=threshold,
                            language=ui_language,
                        )
                        if task == "binary"
                        else regression_confusion_artifact(yte, y_pred, language=ui_language)
                    ),
                    lambda: reliability_artifact(
                        yte,
                        y_pred,
                        task=task,
                        language=ui_language,
                    ),
                    lambda: gradcam_1d_artifact(
                        model,
                        example_vec,
                        feature_names,
                        task=task,
                        language=ui_language,
                    ),
                ],
                stage="building_diagnostics",
                start_percent=74,
                end_percent=87,
            )

            hist = history.history
            if "val_loss" in hist:
                key = "val_loss"
                idx = int(np.argmin(hist["val_loss"]))
            elif task == "binary" and "val_accuracy" in hist:
                key = "val_accuracy"
                idx = int(np.argmax(hist["val_accuracy"]))
            else:
                key = "loss"
                idx = int(np.argmin(hist["loss"]))

            best_epoch = idx
            best_stats = {k: float(hist[k][best_epoch]) for k in hist}
            last_epoch_stats = {k: float(v[-1]) for k, v in hist.items()}

            ctx_frontend = (
                f"{_t(ui_language, 'Modelo', 'Model')}: {_t(ui_language, 'CNN (numérico, tarefa=', 'CNN (numeric, task=')}{task})\n"
                f"filters={filters}, kernel={kernel_size}, dense_units={dense_units}, "
                f"epochs={epochs}, batch_size={batch_size}, seed={seed}, "
                f"threshold={threshold if task == 'binary' else '—'}\n"
                f"Target: {target_col} | #features={len(feature_names)}"
            )

            ctx_train = (
                f"Split: train={len(Xtr)}, test={len(Xte)}, {_t(ui_language, 'val=20% do treino', 'val=20% of training')}\n"
                f"{_t(ui_language, 'Tempo de treino', 'Training time')}: {_fmt_secs(fit_time)} | {_t(ui_language, 'Melhor época', 'Best epoch')} ({key}): {best_epoch}\n"
                f"{_t(ui_language, 'Treino (última época)', 'Training (last epoch)')}: {last_epoch_stats}\n"
                f"{_t(ui_language, 'Treino (melhor época)', 'Training (best epoch)')}: {best_stats}\n"
                f"{_t(ui_language, 'Métricas de TESTE', 'TEST metrics')}: {metrics_out}" +
                "\n\n" + artifacts_text_summary(explainability_artifacts, language=ui_language)
            )

            explanations = _build_explanation_payload(
                style=style,
                objective=objective,
                ctx_frontend=ctx_frontend,
                ctx_train=ctx_train,
                style_free=style_free,
                ui_language=ui_language,
                progress=progress,
            )

            progress.complete()
            return jsonify({
                "detailed": b64_d,
                "summarized": b64_s,
                "explanation": explanations["long"],
                "short_explanation": explanations["short"],
                "explanations": explanations,
                "explainability": artifacts_to_payload(explainability_artifacts),
                "ui_language_used": ui_language,
                **metrics_out
            }), 200

        else:
            return jsonify({"error": "input_type must be text, numeric or image"}), 400

    except Exception as e:
        progress.fail()
        return jsonify({"error": f"CNN processing failed: {e}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)

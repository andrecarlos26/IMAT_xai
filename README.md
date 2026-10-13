# IMAT - Interpretable Model Analysis Tool

IMAT is a web platform for **Explainable Artificial Intelligence (XAI)** that brings together **model training**, **performance metrics**, **visual diagnostics**, and **personalized textual explanations**. It supports MLP, Random Forest, and CNN models through an interactive interface available in **English and Brazilian Portuguese**.

The bundled examples demonstrate memory usage prediction, next-day closing price prediction, and handwritten digit classification. Users can also upload their own compatible datasets and adjust model hyperparameters.

---

## 📂 Project Structure

```text
.
├── app.py                              # Flask application and analysis endpoints
├── requirements_new.txt                # Dependency snapshot included in the archive
├── paths.txt                           # Original deployment notes
├── model_analysis_module/
│   ├── models.py                       # MLP construction and intermediate outputs
│   ├── random_forest.py                # Random Forest training and tree diagrams
│   ├── cnn.py                          # CNN models for text, numeric data, and images
│   ├── metrics.py                      # Regression and binary classification metrics
│   ├── explainability.py               # Feature importance, saliency, and diagnostics
│   ├── explanation.py                  # Prompts and OpenAI textual explanations
│   ├── flowchart.py                    # MLP flowchart generation
│   ├── visualization.py                # Graphviz images encoded as Base64
│   └── utils.py                        # Random seeds and text summary utilities
├── templates/
│   └── index.html                      # HTML/CSS/JavaScript interface served by Flask
└── static/
    ├── datasets/
    │   ├── software_aging_imat.csv      # Default MLP dataset
    │   ├── stock_market_imat.csv        # Default Random Forest dataset
    │   ├── breast_cancer_imat.csv       # Additional binary classification example
    │   └── mnist_imat_sample.npz        # Default CNN image dataset
    ├── favicon.svg
    ├── favicon.ico
    ├── favicon-32x32.png
    └── apple-touch-icon.png
```

---

## ✨ Overview & Goals

1. **Configurable experiments:** Train models with selectable hyperparameters, uploaded datasets, and optional random seeds.
2. **Performance analysis:** Review prediction errors, classification metrics, learning curves, and diagnostic plots.
3. **Model interpretation:** Inspect feature importance, one-dimensional Grad-CAM, token perturbation, and image saliency according to the selected model.
4. **Adaptable explanations:** Choose an explanation style and switch between short and detailed responses in the interface language.
5. **Execution feedback:** Follow processing stages, completed epochs or trees, overall progress, and elapsed time.

IMAT supports exploratory analysis and demonstrations by presenting model results alongside evidence that helps users interpret them.

---

## 🧠 Supported Models & Diagnostics

| Model / input | Task | Main metrics | Interpretation methods |
| --- | --- | --- | --- |
| MLP / numeric CSV | Regression | MAE, MSE, RMSE, R² | SHAP feature importance with permutation fallback |
| Random Forest / numeric CSV | Regression | MAE, MSE, RMSE, R² | Tree SHAP with permutation fallback and tree diagrams |
| CNN / numeric CSV | Regression or binary classification | Regression metrics or accuracy, precision, recall, F1, AUC | Feature importance and Grad-CAM 1D |
| CNN / text CSV | Binary classification | Accuracy, precision, recall, F1, AUC | Grad-CAM 1D and token perturbation |
| CNN / image NPZ | Multiclass classification | Accuracy and macro precision, recall, F1 | Pixel importance from input gradients |

Additional diagnostics include:

- **Learning curves:** Training and validation loss for neural networks.
- **Forest error curves:** Training and test MSE as trees are added to the prediction ensemble.
- **Confusion matrices:** Classification errors, or agreement between discretized value bands for regression.
- **Reliability diagnostics:** Probability calibration for classification and prediction-versus-observation analysis for regression.

**Implementation detail:** The image diagnostic retains the legacy function name `shap_image_artifact` and a SHAP-related label, but its current pixel map is calculated using gradient saliency. It does not compute image SHAP values.

The backend also generates summarized and detailed Graphviz flowcharts. Their display and download controls are currently commented out in the frontend, while the images remain available in API responses.

---

## 🏗️ Modules & Responsibilities

### 1. `app.py`

Orchestrates each analysis:

1. Receives datasets, hyperparameters, language, and explanation preferences.
2. Validates inputs, prepares data, and trains the selected model.
3. Computes metrics, diagnostics, and flowchart images.
4. Requests short and detailed explanations and returns the results as JSON.

It also serves the interface, provides a health endpoint, and stores temporary progress snapshots for browser polling.

### 2. `model_analysis_module/models.py`

- `train_mlp(...)`: Builds and compiles a configurable dense regression network with a linear output. Training is performed by `app.py`.
- `get_intermediate_outputs(model, data)`: Extracts layer outputs for further inspection.

### 3. `model_analysis_module/random_forest.py`

- `train_rf_reg(...)`: Trains a regression forest with an 80/20 train/test split and returns training statistics, including out-of-bag score.
- Supports progress updates based on completed tree batches.
- `create_rf_flowchart_full_reg(...)` and `create_rf_flowchart_summary_reg(...)`: Generate illustrative tree diagrams.

### 4. `model_analysis_module/cnn.py`

- `prepare_text_data(...)`: Cleans text, tokenizes it, pads sequences, and creates a stratified train/test split.
- `build_cnn_text_model(...)`: Builds an embedding and Conv1D network for binary text classification.
- `build_cnn_numeric_model(...)`: Builds a Conv1D network for regression or binary classification.
- `load_image_npz(...)`: Loads image arrays, normalizes pixel values, and adjusts channel dimensions.
- `build_cnn_image_model(...)`: Builds a Conv2D network with pooling, dense layers, dropout, and a softmax output.
- Provides summarized and detailed CNN architecture diagrams.

### 5. `model_analysis_module/metrics.py`

- `compute_metrics(y_true, y_pred)`: Returns `mae`, `mse`, `rmse`, and `r2`.
- `compute_classification_metrics(...)`: Returns binary classification metrics using a configurable probability threshold. AUC is calculated when possible.

Multiclass image metrics are calculated in `app.py`.

### 6. `model_analysis_module/explainability.py`

Produces plots and structured summaries for learning curves, feature importance, confusion matrices, reliability, Grad-CAM 1D, token perturbation, and image saliency.

Tabular explanations use SHAP when available, with permutation importance as a fallback. `artifacts_to_payload(...)` packages diagnostic metadata and Base64 PNG images for the frontend.

### 7. `model_analysis_module/explanation.py`

Builds prompts from the experiment objective, hyperparameters, metrics, and diagnostic summaries, then calls the **OpenAI Responses API** using `requests`.

- **Short mode:** Five topics, approximately 1,000–1,500 characters, with a final recommendations summary and no charts.
- **Detailed mode:** Sanitized HTML text with expandable diagnostic charts attached by the frontend.
- **Styles:** Executive, operational, data science, educational, and custom.
- **Languages:** English and Brazilian Portuguese.

The legacy `get_openrouter_explanation(...)` function delegates to the OpenAI implementation.

### 8. `flowchart.py`, `visualization.py`, and `utils.py`

- `create_flowchart(...)`: Generates MLP diagrams with layer, weight, bias, and activation information.
- `encode_image(dot)`: Renders a Graphviz diagram as a Base64 PNG.
- `set_seed(seed)`: Sets NumPy, TensorFlow, and Python random seeds to support reproducibility.

### 9. `templates/index.html`

Provides model selection, presets, dataset upload and download, CSV previews with the first ten records, and target selection through column headers.

It also handles language confirmation, progress polling, metric formatting, short/detailed explanation switching, expandable charts, and an in-memory result cache for each model. NPZ datasets have no tabular preview.

---

## 📊 Included Datasets & Default Presets

| Dataset | Contents | Default use / suggested target |
| --- | --- | --- |
| `software_aging_imat.csv` | 6,000 records, 10 columns | MLP regression / `Memory_Usage` |
| `stock_market_imat.csv` | 2,484 records, 68 columns | Random Forest regression / `Close_tomorrow` |
| `breast_cancer_imat.csv` | 569 records, 61 columns | Optional numeric CNN binary classification / `diagnosis` |
| `mnist_imat_sample.npz` | 1,437 training images, 360 test images, 10 classes | CNN image classification / labels inside the NPZ |

The interface automatically attaches a dataset when a model preset is selected.

- **MLP:** Layers `48,24,12`, ReLU, Adam, MSE, 20 epochs, batch size 16, seed 42.
- **Random Forest:** 100 trees, maximum depth 4, seed 42.
- **CNN images:** 16 initial filters, kernel size 3, 32 dense units, 3 epochs, batch size 32, seed 42.

The image workflow uses configurable sample limits. Its default limits are **1,200 training-pool images and 300 test images**. Twenty percent of the selected training pool is reserved for validation.

---

## 🚀 How to Run

### Prerequisites

- **Python 3.10** and `pip` for the example environment below.
- **Graphviz** installed as a system program, with `dot` available in PATH.
- A browser with JavaScript enabled.
- An OpenAI API key and network access to generate textual explanations.

TensorFlow is imported when the application starts, including when using Random Forest. A GPU is optional.

### 1) Create a Virtual Environment

Run commands from the directory containing `app.py`.

**Linux / macOS:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows PowerShell:**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install Python Dependencies

The following is a suggested dependency baseline for the existing Keras 2 code. Validate it on your operating system before recording a reproducible environment.

```bash
python -m pip install "tensorflow==2.15.1" "numpy==1.26.4"
python -m pip install "Flask==3.1.0" "flask-cors==5.0.1" "pandas==2.2.3" "scikit-learn==1.6.1" "matplotlib==3.7.5" "graphviz==0.20.3" "requests==2.32.3" "waitress==3.0.2"
python -m pip install "shap==0.44.1"
python -m pip check
```

SHAP is optional at runtime because tabular diagnostics provide a permutation fallback.

**About the included snapshot:** `requirements_new.txt` contains TensorFlow 2.11, Keras 2.11, NumPy 1.26.4, and `tensorflow-intel`, alongside other pinned packages. Review its version and platform compatibility before using it as an installation lockfile. Keep that snapshot separate from the example baseline above.

The baseline uses TensorFlow 2.15.1 to retain Keras 2 APIs. See the official [TensorFlow / Keras compatibility documentation](https://keras.io/getting_started/#tensorflow--keras-2-backwards-compatibility).

### 3) Install Graphviz

Install the system application separately from the Python `graphviz` package.

**Ubuntu / Debian:**

```bash
sudo apt-get update
sudo apt-get install graphviz
```

**macOS with Homebrew:**

```bash
brew install graphviz
```

**Windows:** Use the [Graphviz installer](https://graphviz.org/download/) and add its `bin` directory to PATH.

Verify the installation:

```bash
dot -V
```

Graphviz is still required because the backend generates flowcharts even while their frontend controls are hidden.

### 4) Configure Textual Explanations

The supplied `model_analysis_module/explanation.py` assigns the API key directly in the source. **Remove that inline credential and rotate it before sharing the project.**

To use an environment variable, replace the existing `OPENAI_API_KEY` assignment with:

```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
```

Then configure your own key before starting Flask.

**Linux / macOS:**

```bash
export OPENAI_API_KEY="your-api-key"
```

**Windows PowerShell:**

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

Other settings already read environment variables:

| Variable | Default in the supplied code | Purpose |
| --- | --- | --- |
| `OPENAI_URL` | `https://api.openai.com/v1/responses` | Explanation API endpoint |
| `OPENAI_MODEL` | `gpt-5.4-mini` | Language model identifier |
| `OPENAI_MAX_OUTPUT_TOKENS` | `2600` | Detailed response token limit |
| `OPENAI_TIMEOUT_SECONDS` | `60` | Timeout for each API request |
| `OPENAI_REASONING_EFFORT` | `none` | Reasoning setting |
| `OPENAI_VERBOSITY` | `high` | Detailed response verbosity |
| `PORT` | `5000` | Port used by `python app.py` |

`OPENAI_PROJECT` and `OPENAI_ORGANIZATION` are optional. Short explanations use a separate request capped at 900 output tokens.

After replacing the key assignment, leaving the key unset allows training, metrics, and diagnostics to run with configuration messages in the explanation panels. Errors from a configured API request propagate as processing errors.

### 5) Start the Application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Flask serves both the interface and the API. Use this address to load the default datasets and submit analyses.

Check application health:

```bash
curl http://127.0.0.1:5000/health
```

Expected response:

```json
{"status": "ok"}
```

For a WSGI server without Flask's development debugger, run Waitress from the same directory:

```bash
waitress-serve --listen=127.0.0.1:5000 --threads=4 app:app
```

---

## 🧪 Running an Example

1. Select **MLP**, **Random Forest**, or **CNN**.
2. Keep the automatically attached dataset or upload your own compatible file.
3. For CSV input, open the preview and select the target column. For text CNNs, also set the text column and use binary labels.
4. Review the prefilled hyperparameters. For image CNNs, review the sample limits.
5. Set the experiment objective, interface language, and explanation style.
6. Click **Run** or **Executar**.
7. Follow the progress dialog until processing finishes.
8. Review the metrics and switch between **Short** and **Detailed** explanations. Expand diagnostic charts in detailed mode as needed.

For a numeric CNN classification example, upload `breast_cancer_imat.csv`, select numeric input, choose binary classification, and set `diagnosis` as the target.

---

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Render the interface |
| GET | `/health` | Return application health |
| POST | `/upload` | Train and analyze an MLP |
| POST | `/upload_rf` | Train and analyze a Random Forest |
| POST | `/upload_cnn` | Train and analyze a text, numeric, or image CNN |
| GET | `/training-progress/<progress_id>` | Read the current processing stage and progress |

Analysis requests use `multipart/form-data`. Common fields are `file`, `ui_language`, `objective`, `ex_prompt`, and optionally `ex_prompt_custom` and `seed`.

Model-specific fields include:

- **MLP:** `target_variable`, `layers`, `activation`, `optimizer`, `loss`, `epochs`, `batch_size`.
- **Random Forest:** `target_variable`, `n_estimators`, optional `max_depth`.
- **CNN:** `input_type` (`text`, `numeric`, or `image`), `filters`, `kernel_size`, `dense_units`, `epochs`, `batch_size`, plus input-specific settings such as `target_variable`, `text_column`, `task`, or image sample limits.

Pass `progress_id` as a query parameter to enable polling. It must contain 8–80 letters, digits, underscores, or hyphens.

Successful responses contain task-specific metrics, `explanation`, `short_explanation`, `explanations`, `explainability`, `ui_language_used`, and the Base64 PNG fields `summarized` and `detailed`.

---

## 🗂️ Dataset Format & Processing

**Numeric CSV:** Include a header and a numeric target. The backend selects numeric predictors and excludes the chosen target. MLP and Random Forest drop rows containing missing values.

**Text CSV:** Include a text column and a target containing binary labels `0` and `1`.

**Image NPZ:** Supply `x_train`, `y_train`, `x_test`, and `y_test`, or a combined `X`/`y` or `x`/`y` pair. Optional `class_names` labels can describe the classes. Images must use `(samples, height, width)` or `(samples, height, width, channels)` arrays, with contiguous integer class labels starting at zero.

### Current Split Behavior

| Workflow | Implemented split |
| --- | --- |
| MLP and numeric/text CNN | 80% training pool, 20% test. Validation uses 20% of the training pool, giving approximately 64% train / 16% validation / 20% test. |
| Random Forest | 80% train / 20% test, with out-of-bag scoring |
| Image CNN | Preserves supplied train/test arrays, or creates an 80/20 split, then applies sample limits and reserves 20% of the selected training pool for validation |

The interface displays a 60/20/20 split banner. The table above describes the current backend behavior.

For time series or forecasting experiments, prepare temporal splits and predictor selection outside the current random-split workflow. The stock example includes `TargetUp`, which should be reviewed for target leakage before using the results to evaluate forecasting performance.

---

> **IMAT:** Bringing model training, diagnostics, and explanations into one interface for interpreting machine learning results.


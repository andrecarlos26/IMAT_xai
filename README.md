# IMAT - Interpretable Models Analysis Tool

IMAT is an XAI (Explainable AI) platform designed to **shed light** and **clarify** the decision-making process of machine learning models. With each upload, it generates a **flowchart** illustrating step-by-step how input features are transformed into a memory exhaustion time prediction, accompanied by **performance metrics** .
---

## 📂 Project Structure

```
.
├── frontend/
│   └── index.html                # Interactive UI in HTML/CSS/JS
├── backend/
│   ├── app.py                    # Flask entry point
│   └── model_analysis_module/    # Modeling and XAI logic organized in submodules
│       ├── utils.py
│       ├── models.py
│       ├── metrics.py
│       ├── flowchart.py
│       ├── visualization.py
└── dataseat/
    └── sentiment140.csv        # Example dataset for testing
```

---

## ✨ Overview & Goals

1. **XAI Transparency**: We demystify the inner workings of an MLP by rendering flowcharts that highlight each weight, bias, and activation calculation, exposing the hidden logic of the model.  
2. **Core Metrics**: We compute MAE, MSE, RMSE, and R² to quantify prediction errors, establish approximate confidence intervals, and detect potential biases or systematic behaviors.  

By combining visual artifacts and numerical metrics, IMAT becomes an essential tool for proactive memory resource management in mission-critical environments.

---

## 🏗️ Modules & Responsibilities

### 1. `model_analysis_module`
This package contains all the modeling logic, XAI artifacts generation, and external service integrations.

- **utils.py**  
  Generic helper functions:  
  - `set_seed(seed)`: Fixes NumPy, TensorFlow, and Python `random` seeds to ensure experiment reproducibility.

- **models.py**  
  Encapsulates MLP definition and training:  
  - `train_mlp(input_shape, layers, activation, optimizer, loss, epochs, batch_size, seed)`: Builds and compiles a fully configurable, no-default-layer MLP, perfect for dynamic use cases.  
  - `get_intermediate_outputs(model, data)`: Extracts activations at each MLP layer, crucial for detailing data flow in the flowchart.

- **metrics.py**  
  Regression metrics calculation:  
  - `compute_metrics(y_true, y_pred)`: Returns a dictionary with `mae`, `mse`, `rmse`, and `r2`, enabling insight into prediction accuracy and variability.
  - Facilitates tracking error trends across different training runs or scenarios.

- **flowchart.py**  
  Graphical representation of decision-making:  
  - `create_flowchart(...)`: Creates Graphviz flowcharts in two modes:  
    - **Detailed**: Includes up to N neurons per layer, formulas `z_j = Σ(w*x) + b`, weight/bias lines, and edge labels.  
    - **Summarized**: Shows only the first neurons for a quick overview.  
  - Customizes spacing (`nodesep`, `ranksep`), horizontal layout, and color-coding for multiplication, bias addition, and activation, complete with explanatory legends.
  - When `return_info=True`, it also collects a list of strings describing each node and edge for textual summary.

- **visualization.py**  
  Converting flowcharts to images for JSON transport:  
  - `encode_image(dot)`: Takes a `graphviz.Digraph`, generates an in-memory PNG, and returns a Base64 string ready for APIs or frontends.

### 2. `app.py`
The Flask orchestrator:  
1. Receives and validates CSV uploads and hyperparameters via POST `/upload`.  
2. Trains the MLP and computes performance metrics.  
3. Generates both summarized and detailed flowcharts and compiles a textual summary.  
4. Returns a comprehensive JSON response containing Base64-encoded images and metrics.

### 3. `frontend/index.html`
A lightweight static application:  
- Form for uploading CSV, entering hyperparameters, and choosing language.  
- Dynamic display of performance metrics and flowcharts (with toggles)  
- Image download and new-tab preview capabilities.

### 4. `dataseat/sentiment140.csv`
An example dataset with social network "X" data, ideal for quick local testing and demonstrations.

---

## 🚀 How to Run

### Prerequisites
- Python 3.9+ and `pip`.  
- Graphviz installed on your system (`dot` available in PATH).  
- (Optional) Virtualenv for an isolated environment.

### 1) Install Dependencies (Backend)
```bash
# From the project root, install all dependencies via requirements.txt
pip install -r requirements.txt
```

### 2) Start the Backend API
```bash
cd backend
python app.py
```
The API will be running at `http://127.0.0.1:5000`.

### 3) Launch the Frontend
- Open `frontend/index.html` directly in your browser, or
- Serve statically:
  ```bash
  cd frontend
  python -m http.server 8000
  ```
  Then visit `http://localhost:8000`.

### 4) Testing with the Dataset
- In the frontend form, fill out the fields as follows:
  1. **Select CSV File:** Choose `sentiment140.csv` from the `dataseat/` folder.
  2. **Layers:** Number of neurons per layer, comma-separated (e.g., `64,64`).
  3. **Activation Function:** Choose from `relu`, `sigmoid`, or `tanh`.
  4. **Optimizer:** Select `adam` or `rmsprop`.
  5. **Loss Function:** Choose `mse` or `mae`.
  6. **Epochs:** Number of training epochs (e.g., `10`).
  7. **Batch Size:** Batch size per iteration (e.g., `32`).
  8. **Random Seed:** (Optional) Numeric seed for reproducibility.
  9. **Target Variable:** Column name in the CSV to predict (e.g., `memory_used`).
- Click **Upload** and wait for processing. The results will display:
  - **Metrics:** MAE, MSE, RMSE, R².
  - **Summarized** and **Detailed** flowcharts.

---

> **IMAT**: Making the black box of models transparent and actionable for engineers and system administrators.


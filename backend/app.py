from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from model_analysis_module.models import train_mlp, get_intermediate_outputs
from model_analysis_module.metrics import compute_metrics
from model_analysis_module.flowchart import create_flowchart
from model_analysis_module.visualization import encode_image
from model_analysis_module.explanation import get_openrouter_explanation

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    # 1) File validation
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 2) Reading and preparing data
    df = pd.read_csv(file).dropna()
    numeric = df.select_dtypes(include='number')
    target = request.form.get('target_variable')
    if target not in numeric.columns:
        return jsonify({'error': 'Invalid target variable'}), 400

    X = numeric.drop(columns=[target]).values.astype(float)
    y = numeric[target].values.astype(float)

    # 3) Hyperparameter extraction
    layers     = list(map(int, request.form['layers'].split(',')))
    activation = request.form['activation']
    optimizer  = request.form['optimizer']
    loss       = request.form['loss']
    epochs     = int(request.form['epochs'])
    batch_size = int(request.form['batch_size'])
    seed_param = request.form.get('seed')
    seed       = int(seed_param) if seed_param else None

    # 4) Train the model
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
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # 5) Calculates metrics
    y_pred  = model.predict(X)
    metrics = compute_metrics(y, y_pred)

    # 6) Prepare flowcharts
    X0    = X[0:1]
    pred0 = model.predict(X0)
    Ws, Bs = zip(*(layer.get_weights() for layer in model.layers))

    b64_d, info = create_flowchart(
        model=model,
        prediction=pred0,
        weights=Ws,
        biases=Bs,
        mse=metrics['mse'],
        max_display_neurons=5,
        return_info=True
    )
    b64_s = create_flowchart(
        model=model,
        prediction=pred0,
        weights=Ws,
        biases=Bs,
        mse=metrics['mse'],
        max_display_neurons=3,
        return_info=False
    )
    summary = "\n".join(info)
    print(summary)

    # 7) Return JSON
    return jsonify({
        'detailed':    b64_d,
        'summarized':  b64_s
        **metrics
    }), 200

if __name__ == '__main__':
    app.run(debug=True)

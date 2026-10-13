# model_analysis_module/cnn.py
import re
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling1D, Dense, Dropout, InputLayer

from .utils import set_seed
from .visualization import encode_image

# --- pré-processamento simples (alinha com o exemplo) ---
_url  = re.compile(r'http\S+')
_at   = re.compile(r'@\w+')
_hash = re.compile(r'#\w+')
_digits = re.compile(r'\d+')

def clean_text(text: str) -> str:
    text = _url.sub('', str(text))
    text = _at.sub('', text)
    text = _hash.sub('', text)
    text = _digits.sub('', text)
    return text.lower().strip()

def prepare_text_data(df, text_col, target_col, vocab_size, maxlen, seed=None, test_size=0.2):
    X_text_raw = df[text_col].astype(str).fillna('')
    y = df[target_col].astype(int).values

    X_clean = X_text_raw.apply(clean_text).values
    tok = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tok.fit_on_texts(X_clean)
    seqs = tok.texts_to_sequences(X_clean)
    X = pad_sequences(seqs, maxlen=maxlen, padding='post')

    Xtr, Xte, ytr, yte, raw_tr, raw_te = train_test_split(
        X, y, X_text_raw.values, test_size=test_size, random_state=seed, stratify=y
    )
    return tok, Xtr, Xte, ytr, yte, raw_te

def build_cnn_text_model(vocab_size, embed_dim, filters, kernel_size, dense_units):
    model = Sequential([
        Embedding(vocab_size, embed_dim),
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- CNN para imagens (.npz: x_train/y_train/x_test/y_test ou x/y) ---
def load_image_npz(file_obj, seed=None, test_size=0.2):
    data = np.load(file_obj, allow_pickle=True)
    keys = set(data.files)

    if {"x_train", "y_train", "x_test", "y_test"}.issubset(keys):
        Xtr = data["x_train"]
        ytr = data["y_train"]
        Xte = data["x_test"]
        yte = data["y_test"]
    elif {"X", "y"}.issubset(keys):
        X = data["X"]
        y = data["y"]
        stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y.astype(int))) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )
    elif {"x", "y"}.issubset(keys):
        X = data["x"]
        y = data["y"]
        stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y.astype(int))) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )
    else:
        raise ValueError("Image .npz must contain x_train/y_train/x_test/y_test or X/y arrays")

    class_names = data["class_names"].astype(str).tolist() if "class_names" in keys else None

    Xtr = np.asarray(Xtr, dtype="float32")
    Xte = np.asarray(Xte, dtype="float32")
    ytr = np.asarray(ytr).astype("int64")
    yte = np.asarray(yte).astype("int64")

    if Xtr.max() > 1.0:
        Xtr = Xtr / 255.0
        Xte = Xte / 255.0

    if Xtr.ndim == 3:
        Xtr = Xtr[..., np.newaxis]
        Xte = Xte[..., np.newaxis]

    if Xtr.ndim != 4:
        raise ValueError("Image arrays must have shape (samples, height, width) or (samples, height, width, channels)")

    return Xtr, Xte, ytr, yte, class_names


def build_cnn_image_model(input_shape, num_classes, filters=16, kernel_size=3, dense_units=32):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=filters * 2, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_image_flowchart(model, input_shape, num_classes, dpi=1999, detailed=False):
    dot = graphviz.Digraph(comment='Fluxograma - CNN (Image)')
    dot.graph_attr.update({'rankdir':'LR','dpi':str(dpi),'size':'12,8','nodesep':'1.0','ranksep':'1.2'})
    dot.node('Input', f"Image input\nShape: {input_shape}", shape='box', style='filled', fillcolor='lightgrey')
    prev = 'Input'
    conv_count = dense_count = pool_count = 0
    for i, layer in enumerate(model.layers):
        lid = f'L{i}'
        name = layer.__class__.__name__
        if isinstance(layer, Conv2D):
            conv_count += 1
            dot.node(lid, f"Conv2D #{conv_count}\nFilters: {layer.filters}\nKernel: {layer.kernel_size}\nActivation: relu", shape='ellipse', style='filled', fillcolor='lightgray')
        elif isinstance(layer, MaxPooling2D):
            pool_count += 1
            dot.node(lid, f"MaxPooling2D #{pool_count}\nPool: {layer.pool_size}", shape='ellipse', style='filled', fillcolor='lightgray')
        elif isinstance(layer, Flatten):
            dot.node(lid, 'Flatten', shape='ellipse', style='filled', fillcolor='lightgray')
        elif isinstance(layer, Dense):
            dense_count += 1
            dot.node(lid, f"Dense #{dense_count}\nUnits: {layer.units}\nActivation: {layer.activation.__name__}", shape='ellipse', style='filled', fillcolor='lightgray')
        elif isinstance(layer, Dropout):
            if not detailed:
                continue
            dot.node(lid, f"Dropout\nRate: {layer.rate}", shape='ellipse', style='filled', fillcolor='lightyellow')
        else:
            continue
        dot.edge(prev, lid)
        prev = lid
    dot.node('Output', f"Model output\nClasses: {num_classes}", shape='box', style='filled', fillcolor='lightcoral')
    dot.edge(prev, 'Output', label='softmax prediction')
    return encode_image(dot)

# --- Fluxogramas com design idêntico ao exemplo ---
def _dot_base(title: str, dpi: int = 1999):
    dot = graphviz.Digraph(comment=title)
    dot.graph_attr.update({
        'rankdir':'LR','dpi':str(dpi),'size':'12,8','nodesep':'1.0','ranksep':'1.2'
    })
    return dot

def create_cnn_flowchart_full(model, example_seq, example_text,
                              threshold=0.5, max_filters=1000, dpi=1999):
    dot = _dot_base('Fluxograma - CNN (Detalhado)', dpi=dpi)
    # Entrada
    tokens = example_seq[:10].tolist()
    summary = example_text if len(example_text)<=200 else example_text[:200]+'...'
    dot.node('0', f"Input Sequence\nTokens: {tokens}\nText: {summary}",
             shape='box', style='filled', fillcolor='lightgrey')
    prev = '0'

    # Forward do embedding (para cálculos)
    emb_layer = model.layers[0]
    emb_out = emb_layer(np.expand_dims(example_seq, 0)).numpy()[0]

    for i, layer in enumerate(model.layers):
        lid = f'L{i+1}'
        if isinstance(layer, Embedding):
            dot.node(lid, f"Embedding Layer\nOutput shape: {emb_out.shape}",
                     shape='ellipse', style='filled', fillcolor='lightgray')
            dot.edge(prev, lid)
            prev = lid

        elif isinstance(layer, Conv1D):
            filters, kernel = layer.filters, layer.kernel_size[0]
            act = layer.activation.__name__
            dot.node(lid, f"Conv1D Layer\nFilters: {filters}\nKernel size: {kernel}\nActivation: {act}",
                     shape='ellipse', style='filled', fillcolor='lightgray')
            dot.edge(prev, lid)
            w, b = layer.get_weights()

            # cálculo explícito por filtro (como no exemplo)
            for j in range(min(max_filters, filters)):
                patch = emb_out[:kernel]
                wj = w[:, :, j]
                flat_patch = patch.flatten()
                flat_wj = wj.flatten()
                sum_wx = float(np.dot(flat_patch, flat_wj))
                bias = float(b[j])
                z = sum_wx + bias
                a = float(max(0, z))

                formula = (f"sum(wx) = {sum_wx:.4f}\n"
                           f"bias = {bias:.4f}\n"
                           f"z = {sum_wx:.4f} + {bias:.4f} = {z:.4f}")
                dot.node(f'Calc{i}_{j}', f"Filter {j} calculation:\n{formula}",
                         shape='ellipse', style='filled', fillcolor='lightblue')

                act_color = 'lightgreen' if a > 0 else 'gray'
                dot.node(f'Act{i}_{j}', f"Activation ReLU:\na = max(0, {z:.4f}) = {a:.4f}",
                         shape='ellipse', style='filled', fillcolor=act_color)

                dot.edge(lid, f'Calc{i}_{j}', label='compute z')
                dot.edge(f'Calc{i}_{j}', f'Act{i}_{j}', label=f'{a:.4f}')

            prev = lid

        elif isinstance(layer, GlobalMaxPooling1D):
            dot.node(lid, "Global Max Pooling",
                     shape='ellipse', style='filled', fillcolor='lightgray')
            dot.edge(prev, lid)
            prev = lid

        elif isinstance(layer, Dense):
            units, act = layer.units, layer.activation.__name__
            dot.node(lid, f"Dense Layer\nUnits: {units}\nActivation: {act}",
                     shape='ellipse', style='filled', fillcolor='lightgray')
            dot.edge(prev, lid)
            w, b = layer.get_weights()

            max_units = min(1000, units)
            for j in range(max_units):
                sum_wx = float(np.dot(w[:, j], np.ones(w.shape[0])))
                bias = float(b[j])
                z = sum_wx + bias
                if act == 'sigmoid':
                    a = float(1/(1+np.exp(-z)))
                    formula = (f"sum(wx)= {sum_wx:.4f}\n"
                               f"bias= {bias:.4f}\n"
                               f"z= {sum_wx:.4f}+{bias:.4f}={z:.4f}\n"
                               f"a=1/(1+e^(-{z:.4f}))={a:.4f}")
                else:
                    a = float(max(0, z))
                    formula = (f"sum(wx)= {sum_wx:.4f}\n"
                               f"bias= {bias:.4f}\n"
                               f"z= {sum_wx:.4f}+{bias:.4f}={z:.4f}\n"
                               f"a=max(0, {z:.4f})={a:.4f}")

                dot.node(f'DCalc{i}_{j}', f"Neuron {j} calculation:\n{formula}",
                         shape='ellipse', style='filled', fillcolor='lightblue')
                dot.node(f'DAct{i}_{j}', f"Output a_{j} = {a:.4f}",
                         shape='ellipse', style='filled',
                         fillcolor=('lightgreen' if a>0 else 'gray'))
                dot.edge(lid, f'DCalc{i}_{j}', label='compute z,a')
                dot.edge(f'DCalc{i}_{j}', f'DAct{i}_{j}', label=f'{a:.4f}')
            prev = lid

    # Saída final (mesmo texto do exemplo)
    pred = float(model.predict(np.expand_dims(example_seq, 0)).flatten()[0])
    sentiment = 'Positive' if pred >= threshold else 'Negative'
    dot.node('Output', f"Model Output:\nSentiment: {sentiment}",
             shape='box', style='filled', fillcolor='lightcoral')
    dot.edge(prev, 'Output', label='Prediction')

    # Legenda idêntica
    dot.node('Legend','Legend',shape='plaintext')
    dot.node('M','sum(wx)+bias',shape='box',style='filled',fillcolor='lightblue')
    dot.node('A','Activation ReLU',shape='box',style='filled',fillcolor='lightgreen')
    dot.node('I','Inactive',shape='box',style='filled',fillcolor='gray')
    for n in ['M','A','I']:
        dot.edge('Legend', n)

    return encode_image(dot)

def create_cnn_flowchart_summary(model, example_seq, example_text,
                                 threshold=0.5, dpi=1999):
    # versão enxuta: poucos filtros/unidades para caber bem como "summarized"
    # Reaproveita a full mas com limites mais baixos:
    # para isso, temporariamente "monkey patch" com wrappers pequenos.
    # Simples e eficaz.
    return create_cnn_flowchart_full(
        model, example_seq, example_text,
        threshold=threshold, max_filters=3, dpi=dpi
    )


def build_cnn_numeric_model(input_len, filters, kernel_size, dense_units, task='regression'):
    # (batch, steps, channels=1)
    model = Sequential([
        InputLayer(input_shape=(input_len, 1)),
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(dense_units, activation='relu'),
        Dense(1, activation=('sigmoid' if task=='binary' else 'linear'))
    ])
    if task == 'binary':
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_cnn_flowchart_full_numeric(model, example_vec, feature_names,
                                      task='regression', threshold=0.5,
                                      max_filters=1000, dpi=1999):
    dot = graphviz.Digraph(comment='Fluxograma - CNN (Numeric)')
    dot.graph_attr.update({'rankdir':'LR','dpi':str(dpi),'size':'12,8','nodesep':'1.0','ranksep':'1.2'})

    # Entrada
    label = "\n".join(f"{n}={float(v):.3f}" for n, v in zip(feature_names, example_vec[:min(10,len(example_vec))]))
    dot.node('0', f"Input Vector\n{label}", shape='box', style='filled', fillcolor='lightgrey')
    prev = '0'

    # Conv1D
    conv = model.layers[0]
    w, b = conv.get_weights()  # (kernel, in_ch=1, filters), (filters,)
    filters, kernel = conv.filters, conv.kernel_size[0]
    dot.node('L1', f"Conv1D Layer\nFilters: {filters}\nKernel size: {kernel}\nActivation: relu",
             shape='ellipse', style='filled', fillcolor='lightgray')
    dot.edge(prev, 'L1')

    # cálculos explícitos (patch inicial)
    patch = example_vec[:kernel].reshape(kernel,1)
    for j in range(min(max_filters, filters)):
        wj = w[:,:,j]
        sum_wx = float(np.dot(patch.flatten(), wj.flatten()))
        bias = float(b[j])
        z = sum_wx + bias
        a = max(0.0, z)
        dot.node(f'Calc0_{j}', f"Filter {j} calculation:\nsum(wx) = {sum_wx:.4f}\nbias = {bias:.4f}\nz = {sum_wx:.4f} + {bias:.4f} = {z:.4f}",
                 shape='ellipse', style='filled', fillcolor='lightblue')
        dot.node(f'Act0_{j}', f"Activation ReLU:\na = max(0, {z:.4f}) = {a:.4f}",
                 shape='ellipse', style='filled', fillcolor=('lightgreen' if a>0 else 'gray'))
        dot.edge('L1', f'Calc0_{j}', label='compute z')
        dot.edge(f'Calc0_{j}', f'Act0_{j}', label=f'{a:.4f}')
    prev = 'L1'

    # Global Max Pooling
    dot.node('L2', "Global Max Pooling", shape='ellipse', style='filled', fillcolor='lightgray')
    dot.edge(prev, 'L2'); prev = 'L2'

    # Dense(s)
    dense1 = model.layers[2]
    dot.node('L3', f"Dense Layer\nUnits: {dense1.units}\nActivation: relu",
             shape='ellipse', style='filled', fillcolor='lightgray')
    dot.edge(prev, 'L3'); prev = 'L3'

    out = model.layers[-1]
    act = 'sigmoid' if task=='binary' else 'linear'
    dot.node('L4', f"Dense Layer\nUnits: 1\nActivation: {act}",
             shape='ellipse', style='filled', fillcolor='lightgray')
    dot.edge(prev, 'L4'); prev = 'L4'

    # Saída final
    pred = float(model.predict(example_vec.reshape(1,-1,1), verbose=0).flatten()[0])
    if task == 'binary':
        label_out = f"Model Output:\nClass: {'Positive' if pred>=threshold else 'Negative'}"
    else:
        label_out = f"Model Output:\nPrediction: {pred:.4f}"
    dot.node('Output', label_out, shape='box', style='filled', fillcolor='lightcoral')
    dot.edge(prev, 'Output', label='Prediction')

    # Legenda idêntica ao exemplo
    dot.node('Legend','Legend',shape='plaintext')
    dot.node('M','sum(wx)+bias',shape='box',style='filled',fillcolor='lightblue')
    dot.node('A','Activation ReLU',shape='box',style='filled',fillcolor='lightgreen')
    dot.node('I','Inactive',shape='box',style='filled',fillcolor='gray')
    for n in ['M','A','I']:
        dot.edge('Legend', n)

    return encode_image(dot)

def create_cnn_flowchart_summary_numeric(model, example_vec, feature_names,
                                         task='regression', threshold=0.5, dpi=1999):
    return create_cnn_flowchart_full_numeric(
        model, example_vec, feature_names, task=task, threshold=threshold,
        max_filters=3, dpi=dpi
    )
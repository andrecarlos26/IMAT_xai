import numpy as np
import graphviz
from model_analysis_module.visualization import encode_image
from model_analysis_module.metrics import compute_metrics

def relu(x):
    return max(0, x)

def create_flowchart(model,
                     prediction,
                     weights,
                     biases,
                     mse,
                     max_display_neurons=5,
                     return_info=False):
    """
    Generates the detailed or summarized flowchart (up to max_display_neurons neurons)
    and optionally returns a list of strings with nodes and edges.
    """
    info = []
    dot = graphviz.Digraph(comment='MLP Decision Process')
    dot.graph_attr.update({
        'rankdir': 'LR',
        'dpi': '999',
        'size': '12,8',
        'nodesep': '1.0',
        'ranksep': '1.2'
    })

    # Start node
    dot.node('0', 'Data Input\n[Features]', shape='box')
    if return_info:
        info.append("Node [0]: Data Input [Features] (shape=box)")
    prev = '0'

    # Cycles through layers of the model
    for i, layer in enumerate(model.layers):
        layer_name = layer.name if layer.name != 'Output_Layer' else 'Output Layer'
        act_fn     = layer.activation.__name__
        units      = layer.output_shape[-1]
        W, b       = weights[i], biases[i]

        node_id = f'HL{i+1}'
        dot.node(node_id,
                 f"{layer_name}\nNeurons: {units}\nActivation: {act_fn}",
                 shape='ellipse', style='filled', fillcolor='lightgray')
        dot.edge(prev, node_id, label='Data Flow')
        if return_info:
            info += [
                f"Node [{node_id}]: {layer_name} (ellipse, lightgray)",
                f"Edge: from [{prev}]  to [{node_id}] (label='Data Flow')"
            ]

        # Calculations per neuron
        for j in range(units if return_info else min(units, max_display_neurons)):
            z_val = float(np.dot(W[:, j], np.ones(W.shape[0])) + b[j])
            act_val = relu(z_val) if act_fn=='relu' else z_val
            w_lines = "\n".join(f"w{idx}: {w:.4f}" for idx, w in enumerate(W[:, j][:5]))
            if W.shape[0] > 5:
                w_lines += "\n..."
            b_line = f"b{j}: {b[j]:.4f}"

            mul_id = f"W{i}_{j}"
            dot.node(mul_id,
                     f"Multiplication (z_{j}):\n"
                     f"z_{j} = Σ(w*x) + b = {z_val:.4f}\n"
                     f"{w_lines}\n{b_line}\n\n"
                     f"(Linear Multiplication)",
                     shape='ellipse', style='filled', fillcolor='lightblue')
            bias_id = f"B{i}_{j}"
            dot.node(bias_id,
                     f"Addition of bias:\n{b_line}\n\n"
                     f"Adjusts the activation of the neuron by shifting the calculated value",
                     shape='ellipse', style='filled', fillcolor='lightyellow')
            act_id = f"A{i}_{j}"
            dot.node(act_id,
                     f"Activation Function:\n"
                     f"a{j} = {act_fn}(z_{j}) = {act_val:.4f}\n\n"
                     f"Defines the neuron output after processing",
                     shape='ellipse', style='filled',
                     fillcolor='lightgreen' if act_val>0 else 'gray')

            dot.edge(node_id, mul_id, label=f'Neuron z_{j}')
            dot.edge(mul_id, bias_id, label=f"Output: {z_val:.4f}")
            dot.edge(bias_id, act_id, label=f"Output: {act_val:.4f}")

            if return_info:
                info += [
                    f"Node [{mul_id}]: Multiplication (z_{j}) (ellipse, lightblue)",
                    f"Edge: from [{node_id}]  to [{mul_id}] (label='Neuron z_{j}')",
                    f"Edge: from [{mul_id}]  to [{bias_id}] (label='Output: {z_val:.4f}')",
                    f"Edge: from [{bias_id}]  to [{act_id}] (label='Output: {act_val:.4f}')"
                ]

        prev = node_id

    # Exit node
    pred_val = float(prediction[0][0])
    dot.node('Output',
             f'Model Output (#Time)\nPrediction: {pred_val:.1f}h',
             shape='ellipse', style='filled', fillcolor='lightcoral')
    dot.edge(prev, 'Output', label='Prediction')
    if return_info:
        info += [
            f"Node [Output]: Prediction (ellipse, lightcoral)",
            f"Edge: from [{prev}]  to [Output] (label='Prediction')"
        ]

    # Metrics node
    stats = compute_metrics
    dot.node('Stats',
             f'Model Metrics\nMean Squared Error (MSE): {mse:.8f}',
             shape='box', style='filled', fillcolor='yellow')
    dot.edge('Output', 'Stats', label='Model Confidence')
    if return_info:
        info += [
            f"Node [Stats]: MSE {mse:.8f} (box, yellow)",
            "Edge: from [Output]  to [Stats] (label='Model Confidence')"
        ]

    # Legend
    legend_items = [
        ("Multiplication","box","lightblue"),
        ("Bias",       "box","lightyellow"),
        ("Activated",  "box","lightgreen"),
        ("Inactive",   "box","gray")
    ]
    dot.node("Legend","Legend",shape='plaintext')
    if return_info:
        info.append("Node [Legend]: Legend (plaintext)")
    for tag, shape, color in legend_items:
        dot.node(tag, tag, shape=shape, style='filled', fillcolor=color)
        dot.edge("Legend", tag)
        if return_info:
            info.append(f"Node [{tag}]: {tag} ({shape}, {color})")
            info.append(f"Edge: from [Legend]  to [{tag}]")

    image_b64 = encode_image(dot)
    return (image_b64, info) if return_info else image_b64

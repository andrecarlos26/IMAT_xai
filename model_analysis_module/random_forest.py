import numpy as np
import time
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from graphviz import Digraph
from .visualization import encode_image

def train_rf_reg(
    df,
    target,
    n_estimators,
    max_depth,
    seed,
    progress_callback=None,
):
    numeric = df.select_dtypes(include='number')
    features = numeric.drop(columns=[target]).values.astype(float)
    y       = numeric[target].values.astype(float)
    Xtr, Xte, ytr, yte = train_test_split(
        features, y, test_size=0.2, random_state=seed
    )
    if n_estimators < 1:
        raise ValueError("n_estimators must be greater than zero")

    # When progress reporting is enabled, warm_start lets us train the forest
    # in deterministic batches. Each update therefore reflects trees that have
    # actually finished training instead of a timer-based estimate.
    progress_batches = min(20, n_estimators) if progress_callback else 1
    batch_size = int(np.ceil(n_estimators / progress_batches))
    initial_estimators = min(batch_size, n_estimators)

    rf = RandomForestRegressor(
        n_estimators=initial_estimators,
        max_depth=max_depth,
        random_state=seed,
        bootstrap=True,
        # OOB is calculated once after the final progress batch.
        oob_score=not bool(progress_callback),
        warm_start=bool(progress_callback),
    )
    t0 = time.time()

    if progress_callback:
        trained = 0
        progress_callback(trained, n_estimators)
        while trained < n_estimators:
            trained = min(trained + batch_size, n_estimators)
            rf.set_params(n_estimators=trained)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                rf.fit(Xtr, ytr)
            progress_callback(trained, n_estimators)

        # Reusing the fully trained trees computes the final OOB score without
        # retraining them or distorting the reported tree progress.
        rf.set_params(oob_score=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rf.fit(Xtr, ytr)
        rf.set_params(warm_start=False)
    else:
        rf.fit(Xtr, ytr)

    fit_time = time.time() - t0

    depths = [est.tree_.max_depth for est in rf.estimators_]
    nodes  = [est.tree_.node_count for est in rf.estimators_]
    stats = {
        'fit_time': fit_time,
        'oob_score': getattr(rf, 'oob_score_', None),
        'avg_depth': float(np.mean(depths)) if depths else float('nan'),
        'avg_nodes': float(np.mean(nodes))  if nodes  else float('nan'),
    }
    return rf, Xtr, Xte, ytr, yte, stats

def create_rf_flowchart_full_reg(
    rf_model, example_vec, feature_names, max_trees=5,
    max_depth=None, dpi=1999
):
    dot = Digraph(comment='RF Reg – completo', strict=True)
    dot.graph_attr.update(rankdir='LR', size='16,10', dpi=str(dpi))

    # nó de input
    label = "\n".join(f"{n}={v:.2f}"
                      for n, v in zip(feature_names, example_vec))
    dot.node('Input', f"Input\n{label}",
             shape='box', style='filled', fillcolor='lightgrey')

    tree_preds = []
    for i, est in enumerate(rf_model.estimators_[:max_trees]):
        tree = est.tree_
        cluster = f'cluster{i}'
        with dot.subgraph(name='cluster_'+cluster) as c:
            c.attr(label=f"Árvore {i+1}", color='black')

            def recurse(node, depth, parent=None, edge_label=''):
                name = f"{cluster}_n{node}"
                if tree.children_left[node] == -1:
                    # folha: valor médio da predição
                    pred = float(est.predict([example_vec])[0])
                    c.node(name, f"Leaf#{node}\npred={pred:.2f}",
                           shape='box', style='filled', fillcolor='lightcoral')
                else:
                    fidx = tree.feature[node]
                    thr  = tree.threshold[node]
                    val  = example_vec[fidx]
                    direction = 'True' if val<=thr else 'False'
                    c.node(name, f"{feature_names[fidx]}\nval={val:.2f}\nthr={thr:.2f}\n→{direction}",
                           shape='ellipse', style='filled', fillcolor='lightblue')
                if parent:
                    c.edge(parent, name, label=edge_label)
                if tree.children_left[node] != -1 and (max_depth is None or depth<max_depth):
                    recurse(tree.children_left[node], depth+1, parent=name, edge_label='True')
                    recurse(tree.children_right[node], depth+1, parent=name, edge_label='False')

            recurse(0, 0)
            leaf_pred = float(est.predict([example_vec])[0])
            tree_preds.append(leaf_pred)
            vote_name = f"{cluster}_vote"
            c.node(vote_name, f"Predição Árvore {i+1}\n{leaf_pred:.2f}",
                   shape='box', style='filled', fillcolor='lightsteelblue')
            c.edge(f"{cluster}_n0", vote_name, label='→')

        dot.edge('Input', f"{cluster}_n0", label=f"Árvore {i+1}")

    # voto final = média das predições
    final_pred = float(np.mean(tree_preds))
    dot.node('Final', f"Predição Final\n{final_pred:.2f}",
             shape='box', style='filled', fillcolor='lightcoral')
    for i in range(len(tree_preds)):
        dot.edge(f"cluster{i}_vote", 'Final', label='→')

    # legenda
    dot.node('Legend', 'Legenda', shape='plaintext')
    items = [
        ('In','box','lightgrey'),
        ('Dec','ellipse','lightblue'),
        ('Lf','box','lightcoral'),
        ('Vt','box','lightsteelblue'),
        ('Fv','box','lightcoral')
    ]
    for tag, shape, color in items:
        dot.node(tag, tag, shape=shape, style='filled', fillcolor=color)
        dot.edge('Legend', tag)

    return encode_image(dot)

def create_rf_flowchart_summary_reg(
    rf_model, example_vec, feature_names
):
    # resumo: 1 árvore, profundidade 1
    return create_rf_flowchart_full_reg(
        rf_model, example_vec, feature_names,
        max_trees=1, max_depth=1, dpi=1999
    )

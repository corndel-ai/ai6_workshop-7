"""Helper functions for Workshop 7: Production Grade Classification.

The notebook imports everything it needs from this file, so keep it in the
same folder as the notebook.

Plots
    plot_class_distribution, plot_confusion_matrix, plot_roc_calibration,
    threshold_table, plot_shap_values

Run tracking
    configure_tracking, log_run, show_runs, wandb_project_url,
    register_model

Modelling and reporting
    train_clean_model, compare_models, describe_change, evaluate, risk_tiers, pick_profiles,
    save_model_meta, example_payload, create_model_card, card_numbers

Success checks
    check_target, check_split, check_recall, check_pos_weight,
    check_threshold, check_protected, check_features, check_best_run,
    check_model_card, check_what_if, check_gate, check_slides
"""

import json
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

_BLUE = '#2a6ebb'
_RED = '#c0392b'
_GREEN = '#27ae60'
_DARK = '#1a1a2e'
_GREY = '#666666'

RUNS_FILE = 'runs_log.csv'
_STATE = {'mode': 'local', 'project': 'ai6-7w-hr-attrition'}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_class_distribution(df, target='Attrition'):
    """Bar chart of how many employees left and how many stayed."""
    if set(df[target].dropna().unique()).issubset({'Yes', 'No'}):
        left, stayed = (df[target] == 'Yes').sum(), (df[target] == 'No').sum()
    else:
        left, stayed = (df[target] == 1).sum(), (df[target] == 0).sum()
    rate = left / len(df)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    bars = ax.bar(['Left', 'Stayed'], [left, stayed],
                  color=[_RED, _BLUE], width=0.5, alpha=0.9)
    for bar, val in zip(bars, [left, stayed]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Employees')
    ax.set_title(f'Class distribution: {rate:.1%} attrition rate',
                 fontsize=12, fontweight='bold')
    plt.show()
    plt.close(fig)
    print(f'For every employee who left, {round((1 - rate) / rate)} stayed.')


def plot_confusion_matrix(y_true, y_pred, title='Model'):
    """Confusion matrix. The red border marks missed attrition."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    labels = [['True negative\n(correctly cleared)',
               'False positive\n(unnecessary flag)'],
              ['False negative\n(missed attrition)',
               'True positive\n(attrition caught)']]
    colours = [['#f0f7f0', '#e8f4fd'], ['#fdecea', '#e8f8f0']]
    values = [[tn, fp], [fn, tp]]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle([j - 0.5, i - 0.5], 1, 1,
                                       linewidth=2, edgecolor='#cccccc',
                                       facecolor=colours[i][j]))
            if i == 1 and j == 0:
                ax.add_patch(plt.Rectangle([j - 0.5, i - 0.5], 1, 1,
                                           linewidth=3, edgecolor=_RED,
                                           facecolor='none', zorder=5))
            colour = (_RED if (i, j) == (1, 0) else
                      _GREEN if (i, j) == (1, 1) else _DARK)
            ax.text(j, i, str(values[i][j]), ha='center', va='center',
                    fontsize=22, fontweight='bold', color=colour)
            ax.text(j, i + 0.32, labels[i][j], ha='center', va='center',
                    fontsize=8, color=_GREY)

    total = tn + fp + fn + tp
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nstayed', 'Predicted\nleft'], fontsize=9)
    ax.set_yticklabels(['Actually\nstayed', 'Actually\nleft'], fontsize=9)
    ax.set_title(f'{title}\nAccuracy {(tp + tn) / total:.1%}  |  '
                 f'Precision {prec:.1%}  |  Recall {rec:.1%}  |  '
                 f'F1 {f1:.3f}', fontsize=11, pad=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.show()
    plt.close(fig)


def plot_roc_calibration(y_true, y_prob):
    """ROC curve with the AUC and a reference line for a random model."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot(fpr, tpr, color=_BLUE, linewidth=2,
            label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#cccccc', linestyle='--',
            label='Random model (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.08, color=_BLUE)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.show()
    plt.close(fig)
    print(f'ROC AUC: {auc:.3f} (0.50 is no better than a coin flip)')


def threshold_table(y_true, y_prob, thresholds=None):
    """Return precision, recall, F1 and error counts at a range of cut-offs."""
    thresholds = thresholds or [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        rows.append({'Threshold': t, 'Precision': round(prec, 3),
                     'Recall': round(rec, 3), 'F1': round(f1, 3),
                     'False alarms (FP)': fp, 'Missed (FN)': fn,
                     'Caught (TP)': tp})
    table = pd.DataFrame(rows).set_index('Threshold')
    return table


def plot_shap_values(model, X_val, max_display=15):
    """SHAP summary plot plus a ranked list of the strongest features."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    shap.summary_plot(shap_values, X_val, max_display=max_display,
                      show=False)
    plt.title('SHAP: contribution of each feature to the attrition '
              'prediction', fontsize=11)
    plt.gcf().set_size_inches(9, 6)
    plt.tight_layout()
    plt.show()
    plt.close('all')
    ranking = pd.Series(np.abs(shap_values).mean(axis=0),
                        index=X_val.columns).sort_values(ascending=False)
    return ranking.head(max_display).round(4).to_frame('mean |SHAP|')


# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------

def configure_tracking(project, mode='wandb'):
    """Choose where runs are recorded.

    Every run is always saved to a local file (runs_log.csv). In 'wandb'
    mode each run is also sent to your W&B dashboard. If login fails or is
    cancelled, tracking falls back to local only and the notebook carries on.
    """
    _STATE['project'] = project
    _STATE['mode'] = 'local'
    if mode == 'wandb':
        try:
            import wandb
            if os.environ.get('WANDB_MODE') in ('offline', 'disabled'):
                _STATE['mode'] = 'wandb'
            elif wandb.login():
                _STATE['mode'] = 'wandb'
        except Exception as error:
            print(f'W&B login did not complete ({type(error).__name__}).')
    if os.path.exists(RUNS_FILE):
        os.remove(RUNS_FILE)
    where = {'wandb': 'W&B and runs_log.csv', 'local': 'runs_log.csv only'}
    print(f'Tracking mode: {_STATE["mode"]}. Runs are saved to '
          f'{where[_STATE["mode"]]}.')


def log_run(run_name, model, X_val, y_val, y_pred, config=None):
    """Record one model run: metrics, confusion counts and the settings."""
    config = config or {}
    y_prob = model.predict_proba(X_val)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1_attrition': f1_score(y_val, y_pred, zero_division=0),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'true_positives': int(tp), 'false_positives': int(fp),
        'true_negatives': int(tn), 'false_negatives': int(fn),
    }
    _save_local(run_name, metrics, config)
    if _STATE['mode'] == 'wandb':
        _send_to_wandb(run_name, model, X_val, metrics, config)
    print(f'Run {run_name} logged. Accuracy {metrics["accuracy"]:.1%}, '
          f'F1 {metrics["f1_attrition"]:.3f}, '
          f'missed departures (FN) {fn}, false alarms (FP) {fp}.')


def _save_local(run_name, metrics, config):
    row = {'run': run_name, **{k: round(v, 4) if isinstance(v, float) else v
                               for k, v in metrics.items()},
           **{k: str(v) for k, v in config.items()}}
    table = (pd.read_csv(RUNS_FILE) if os.path.exists(RUNS_FILE)
             else pd.DataFrame())
    if 'run' in table.columns:
        table = table[table['run'] != run_name]
    pd.concat([table, pd.DataFrame([row])], ignore_index=True).to_csv(
        RUNS_FILE, index=False)


def _send_to_wandb(run_name, model, X_val, metrics, config):
    import wandb
    run = wandb.init(project=_STATE['project'], name=run_name, config=config,
                     settings=wandb.Settings(silent=True))
    wandb.log(metrics)
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_,
                               index=X_val.columns).sort_values(
                                   ascending=False)
        wandb.log({'feature_importance': wandb.Table(
            columns=['feature', 'importance'],
            data=[[f, round(float(v), 5)] for f, v in importance.items()])})
    run.finish()


def show_runs():
    """Return the runs logged so far as a table."""
    columns = ['run', 'accuracy', 'f1_attrition', 'recall', 'roc_auc',
               'false_negatives', 'false_positives', 'threshold']
    table = pd.read_csv(RUNS_FILE)
    return table[[c for c in columns if c in table.columns]].set_index('run')


def wandb_project_url():
    """Return the dashboard address, or explain why there is none."""
    if _STATE['mode'] != 'wandb':
        return 'No W&B dashboard in this session. Your runs are in runs_log.csv.'
    try:
        import wandb
        entity = wandb.Api().default_entity
        return f'https://wandb.ai/{entity}/{_STATE["project"]}'
    except Exception:
        return ('Open wandb.ai and select the project '
                f'{_STATE["project"]} in your account.')


def register_model(model, path, metadata):
    """Save the model and, in wandb mode, register it as a versioned artefact."""
    import joblib
    joblib.dump(model, path)
    print(f'Model saved to {path}')
    if _STATE['mode'] != 'wandb':
        print('Local mode: the saved file is your model version. '
              'No registry entry was created.')
        return
    import wandb
    run = wandb.init(project=_STATE['project'], name='05_registered_model',
                     settings=wandb.Settings(silent=True))
    artifact = wandb.Artifact('hr-attrition-classifier', type='model',
                              metadata=metadata)
    artifact.add_file(path)
    run.log_artifact(artifact)
    run.finish()
    print('Model registered as a W&B artefact.')


def save_model_meta(path, threshold, features, removed):
    """Write the settings app.py needs so the API always matches the model."""
    with open(path, 'w') as handle:
        json.dump({'threshold': threshold, 'features': list(features),
                   'features_removed': list(removed)}, handle, indent=2)
    print(f'Settings saved to {path}')


# ---------------------------------------------------------------------------
# Modelling and reporting
# ---------------------------------------------------------------------------

def train_clean_model(X_train, X_val, y_train, y_val,
                      features_to_remove, pos_weight, threshold=0.3):
    """Train with the chosen features removed. Returns model, X_val, y_pred."""
    import xgboost as xgb
    missing = [f for f in features_to_remove if f not in X_train.columns]
    if missing:
        print(f'Skipped (not found): {missing}')
    keep = [f for f in features_to_remove if f in X_train.columns]
    X_train_clean = X_train.drop(columns=keep)
    X_val_clean = X_val.drop(columns=keep)
    model = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42,
                              eval_metric='logloss', verbosity=0)
    model.fit(X_train_clean, y_train)
    y_pred = (model.predict_proba(X_val_clean)[:, 1] >= threshold)
    print(f'Features removed: {keep if keep else "none"}. '
          f'Features remaining: {X_val_clean.shape[1]}.')
    return model, X_val_clean, y_pred.astype(int)


def risk_tiers(y_prob, threshold):
    """Split flagged employees into three tiers above the threshold."""
    edges = np.linspace(threshold, 1.0, 4)
    flagged = y_prob >= threshold
    tiers = pd.cut(y_prob[flagged], bins=edges, include_lowest=True,
                   labels=['Medium-Low', 'Medium-High', 'High'])
    counts = tiers.value_counts().reindex(['High', 'Medium-High',
                                           'Medium-Low'])
    table = counts.to_frame('Employees')
    table['Probability range'] = ['{:.2f} to {:.2f}'.format(edges[i], edges[i + 1])
                                  for i in (2, 1, 0)]
    table.loc['Not flagged'] = [int((~flagged).sum()),
                                f'below {threshold:.2f}']
    return table


def pick_profiles(model, X_val, y_val, threshold):
    """Pick three real validation rows: lowest, closest to threshold, highest."""
    prob = pd.Series(model.predict_proba(X_val)[:, 1], index=X_val.index)
    picks = {'Lowest risk': prob.idxmin(),
             'Closest to your threshold': (prob - threshold).abs().idxmin(),
             'Highest risk': prob.idxmax()}
    table = pd.DataFrame({
        'Row': list(picks.values()),
        'Predicted probability': [round(prob[i], 3) for i in picks.values()],
        'Actually left': ['Yes' if y_val.loc[i] == 1 else 'No'
                          for i in picks.values()],
    }, index=list(picks.keys()))
    return picks, table


# ---------------------------------------------------------------------------
# Success checks. Each prints a result and never raises, so the notebook
# keeps running if a task is not finished yet.
# ---------------------------------------------------------------------------

def _report(ok, passed, hint):
    print(f'Success check passed: {passed}' if ok else f'Not yet: {hint}')


def check_target(df):
    column = df['Attrition']
    ok = (column.dtype.kind in 'iu' and set(column.unique()) == {0, 1})
    return _report(ok, f'Attrition is coded 0 and 1 ({int(column.sum())} '
                   'leavers).' if ok else '',
                   'every row should be the integer 0 (stayed) or 1 (left), '
                   'with no gaps.')


def check_split(X, y, X_train, y_train):
    from sklearn.model_selection import train_test_split
    expected = train_test_split(X, y, test_size=0.2, random_state=42,
                                stratify=y)[0]
    ok = X_train.index.equals(expected.index)
    return _report(ok, 'the training and test sets keep the same share of leavers.',
                   'add the argument that keeps the leaver share equal in both sets.')


def check_pos_weight(pos_weight, y_train):
    expected = (y_train == 0).sum() / (y_train == 1).sum()
    ok = pos_weight is not None and abs(float(pos_weight) - expected) < 1e-9
    return _report(ok, f'{float(pos_weight):.2f} stayed for every one who left.'
                   if ok else '', 'the ratio is stayed divided by left, '
                   'and it should be above 1.')


def check_threshold(y_prob, y_pred, threshold, reason=''):
    expected = (y_prob >= threshold).astype(int)
    if not (np.asarray(y_pred).dtype.kind in 'iu'
            and np.array_equal(np.asarray(y_pred), expected)):
        return _report(False, '', 'predictions should be integers 0 and 1, '
                       'flagged when the probability reaches your threshold.')
    if not str(reason).strip():
        return _report(False, '', 'write your reasoning in THRESHOLD_REASON.')
    return _report(True, f'predictions use your threshold of {threshold}, '
                   'with a written reason.', '')


def check_recall(recall, y_val, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    ok = recall is not None and abs(float(recall) - tp / (tp + fn)) < 1e-9
    return _report(ok, f'recall is {float(recall):.1%}: {tp} of {tp + fn} '
                   'leavers were caught.' if ok else '',
                   'recall is true positives divided by true positives plus '
                   'false negatives.')


def check_protected(found, ranking):
    expected = [f for f in ['Age', 'Gender', 'MaritalStatus']
                if f in ranking.index]
    ok = found is not None and sorted(found) == sorted(expected)
    return _report(ok, f'protected features in the top {len(ranking)}: '
                   f'{expected if expected else "none"}.',
                   'list every protected feature that appears in the '
                   'ranking, and none that do not.')


def check_best_run(best_run):
    runs = show_runs()
    ok = best_run in runs.index and (
        runs.loc[best_run, 'false_negatives']
        == runs['false_negatives'].min())
    return _report(ok, f'{best_run} has the fewest missed departures '
                   f'({int(runs["false_negatives"].min())}).',
                   'find the row with the smallest false_negatives value.')


def check_model_card(path):
    if not os.path.exists(path):
        return _report(False, '', f'{path} does not exist yet. Run the cell '
                       'that creates it.')
    text = open(path, encoding='utf-8').read()
    empty_cells = len(re.findall(r'\|\s*\|\s*$', text, flags=re.M))
    blank_sections = len(re.findall(r'\*\*[^*\n]+:\*\*\n\n\n', text))
    perf = text.split('## Performance')[1].split('---')[0]
    perf_empty = sum(1 for line in perf.splitlines()
                     if line.count('|') >= 3 and line.strip().endswith('| |')
                     and 'Metric' not in line)
    if perf_empty:
        return _report(False, '', f'{perf_empty} row(s) of the Performance '
                       'table are still empty.')
    if blank_sections:
        return _report(False, '', f'{blank_sections} written section(s) are '
                       'still blank.')
    return _report(True, 'the Performance table and the written sections are '
                   'complete. Ask a peer to read it for gaps.', '')


def check_features(columns, removed, reason):
    unknown = [f for f in removed if f not in columns]
    if unknown:
        return _report(False, '', f'these names are not columns: {unknown}.')
    if not reason.strip():
        return _report(False, '', 'write your reasoning in FEATURE_REASON.')
    return _report(True, f'{len(removed)} feature(s) removed, with a written reason.', '')


def check_what_if(row, feature, new_value):
    if feature not in row.columns:
        return _report(False, '', f'choose a feature from {list(row.columns)}.')
    old = row[feature].iloc[0]
    return _report(new_value is not None and new_value != old,
                   f'{feature} changed from {old} to {new_value}.',
                   'choose a value that differs from the original.')


def check_gate(gate):
    keys = {'check', 'owner', 'halt_condition'}
    ok = (len(gate) == 3 and all(keys <= set(g) for g in gate)
          and all(len(str(g['check']).split()) >= 3
                  and len(str(g['halt_condition']).split()) >= 3
                  and len(str(g['owner']).split()) >= 1 for g in gate))
    return _report(ok, 'three checks, each with an owner and a halt condition.',
                   'give exactly three checks. Each needs a check and a '
                   'halt_condition of at least three words, and an owner.')


def check_slides(slides):
    keys = ['problem', 'approach', 'headline_result', 'metric_choice',
            'threshold', 'features_removed', 'output_format', 'oversight',
            'accountability']
    empty = [k for k in keys if len(str(slides.get(k, '')).split()) < 3]
    return _report(not empty, 'all nine slide fields are written.',
                   f'write at least three words for: {empty}.')


# ---------------------------------------------------------------------------
# Evaluation and documentation helpers
# ---------------------------------------------------------------------------

def evaluate(y_true, y_prob, threshold):
    """Return the headline metrics at one threshold as a Series."""
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return pd.Series({
        'Accuracy': accuracy_score(y_true, pred),
        'Precision': precision_score(y_true, pred, zero_division=0),
        'Recall': recall_score(y_true, pred, zero_division=0),
        'F1': f1_score(y_true, pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_true, y_prob),
        'Missed (FN)': int(fn), 'False alarms (FP)': int(fp)})


def compare_models(y_val, models):
    """Compare models side by side. models = {name: (model, X, y_pred)}."""
    columns = {}
    for name, (model, X, y_pred) in models.items():
        prob = model.predict_proba(X)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred,
                                          labels=[0, 1]).ravel()
        columns[name] = {
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred, zero_division=0),
            'Recall': recall_score(y_val, y_pred, zero_division=0),
            'F1': f1_score(y_val, y_pred, zero_division=0),
            'ROC AUC': roc_auc_score(y_val, prob),
            'Missed (FN)': int(fn), 'False alarms (FP)': int(fp)}
    return pd.DataFrame(columns).round(3)


def describe_change(table, before, after):
    """Say in one sentence how accuracy and recall moved between two models."""
    def move(metric):
        delta = (table.loc[metric, after] - table.loc[metric, before]) * 100
        if abs(delta) < 0.05:
            return f'{metric.lower()} did not change'
        direction = 'rose' if delta > 0 else 'fell'
        return f'{metric.lower()} {direction} by {abs(delta):.1f} points'
    print(f'{after} compared with {before}: {move("Accuracy")}, '
          f'{move("Recall")}.')


def example_payload(row):
    """Return one employee as JSON, ready to paste into the API docs page."""
    return json.dumps({k: float(v) for k, v in row.items()}, indent=2)


def create_model_card(template, path):
    """Copy the template once. An existing card is never overwritten."""
    if os.path.exists(path):
        print(f'{path} already exists, so it was left as it is.')
        return
    shutil.copy(template, path)
    print(f'Created {path}. Open it in the file explorer to complete it.')


def card_numbers(y_val, model, X_val, threshold):
    """Return the figures for the model card Performance table."""
    scores = evaluate(y_val, model.predict_proba(X_val)[:, 1], threshold)
    table = scores.drop(['Accuracy']).round(3).to_frame('Value')
    table.loc['Decision threshold'] = threshold
    return table

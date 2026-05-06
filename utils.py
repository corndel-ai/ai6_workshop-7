# =============================================================================
#
#   🏢  Unit 7W : Production-Grade Classification in Action
#   📦  Helper functions for the HR attrition notebook
#
# =============================================================================
#
#   Contents
#   ────────
#   📊  plot_confusion_matrix(y_true, y_pred, title)
#           Colour-coded confusion matrix with labelled cells.
#
#   🔍  plot_shap_values(model, X_test)
#           SHAP beeswarm summary plot showing feature contributions.
#
#   📈  plot_roc_calibration(y_true, y_prob)
#           ROC curve with AUC annotation.
#
#   🎯  plot_threshold_table(y_true, y_prob)
#           Precision, recall, F1, FP and FN at key thresholds.
#
#   📡  log_run(project, run_name, model, X_test, y_test, y_pred, config)
#           Log a model run to WandB with standard metrics.
#
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import shap
import wandb
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, roc_curve, precision_score, recall_score,
)

# ── Palette ───────────────────────────────────────────────────────────────────
_BLUE  = '#2a6ebb'
_RED   = '#c0392b'
_GREEN = '#27ae60'
_AMBER = '#e67e22'
_DARK  = '#1a1a2e'
_GREY  = '#666666'


# ── 📊 Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title='Model'):
    """
    Colour-coded confusion matrix. Red = false negative (missed attrition).
    Green = true positive (caught attrition).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    labels = [
        ['True Negative\n(correctly cleared)', 'False Positive\n(unnecessary flag)'],
        ['False Negative\n(missed attrition)',  'True Positive\n(attrition caught)'],
    ]
    colours = [
        ['#f0f7f0', '#e8f4fd'],
        ['#fdecea', '#e8f8f0'],
    ]
    values = [[tn, fp], [fn, tp]]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle(
                [j - 0.5, i - 0.5], 1, 1,
                linewidth=2, edgecolor='#cccccc',
                facecolor=colours[i][j],
            )
            ax.add_patch(rect)
            if i == 1 and j == 0:
                rect_fn = plt.Rectangle(
                    [j - 0.5, i - 0.5], 1, 1,
                    linewidth=3, edgecolor=_RED,
                    facecolor='none', zorder=5,
                )
                ax.add_patch(rect_fn)
            val_colour = (
                _RED   if (i == 1 and j == 0) else
                _GREEN if (i == 1 and j == 1) else
                _DARK
            )
            ax.text(j, i, str(values[i][j]),
                    ha='center', va='center',
                    fontsize=22, fontweight='bold', color=val_colour)
            ax.text(j, i + 0.32, labels[i][j],
                    ha='center', va='center', fontsize=8, color=_GREY)

    acc  = (tp + tn) / cm.sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nstayed', 'Predicted\nleft'], fontsize=9)
    ax.set_yticklabels(['Actually\nstayed', 'Actually\nleft'], fontsize=9)
    ax.set_title(
        f'{title}\n'
        f'Acc {acc:.1%}  |  Prec {prec:.1%}  |  Rec {rec:.1%}  |  F1 {f1:.3f}',
        fontsize=11, pad=10,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.5, -0.02,
        'Red border = false negative (missed attrition).',
        ha='center', fontsize=9, color=_RED, style='italic',
    )
    plt.tight_layout()
    plt.show()


# ── 🔍 SHAP values ────────────────────────────────────────────────────────────

def plot_shap_values(model, X_test, max_display=15):
    """
    SHAP beeswarm summary plot. Shows which features drive predictions
    and in which direction.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_test,
        max_display=max_display,
        show=False,
    )
    plt.title('SHAP feature importance : contribution to attrition prediction',
              fontsize=12, pad=12)
    plt.tight_layout()
    plt.show()

    # Also print ranked list
    mean_abs = np.abs(shap_values).mean(axis=0)
    import pandas as pd
    ranking = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=False)
    print('\nFeature importance (mean |SHAP|):')
    print(ranking.head(15).round(4).to_string())


# ── 📈 ROC curve ──────────────────────────────────────────────────────────────

def plot_roc_calibration(y_true, y_prob):
    """
    ROC curve with AUC annotation and a reference line at 0.5.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=_BLUE, linewidth=2,
            label=f'ROC curve  (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#cccccc', linewidth=1,
            linestyle='--', label='Random classifier (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.08, color=_BLUE)

    ax.set_xlabel('False positive rate', fontsize=11)
    ax.set_ylabel('True positive rate', fontsize=11)
    ax.set_title('ROC curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    print(f'\nROC AUC: {auc:.3f}')
    print('(0.50 = random, 0.63–0.70 = weak signal, 0.70–0.80 = moderate)')


# ── 🎯 Threshold table ────────────────────────────────────────────────────────

def plot_threshold_table(y_true, y_prob, thresholds=None):
    """
    Print precision, recall, F1, false positives and false negatives
    at a range of thresholds.
    """
    if thresholds is None:
        thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]

    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} "
          f"{'F1':<10} {'FP':<8} {'FN':<8} {'TP':<6}")
    print('-' * 70)

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm    = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f'{t:<12.2f} {prec:<12.3f} {rec:<12.3f} {f1:<10.3f} '
              f'{fp:<8} {fn:<8} {tp:<6}')

    print('\nFP = false alarms  |  FN = missed attrition  |  TP = correctly flagged')


# ── 📡 WandB logging ──────────────────────────────────────────────────────────

def log_run(project, run_name, model, X_test, y_test, y_pred, config=None):
    """
    Log a model run to WandB with standard classification metrics
    and feature importance.
    """
    import pandas as pd

    config = config or {}

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        reinit=True,
    )

    wandb.log({
        'accuracy':         acc,
        'f1_attrition':     f1,
        'precision':        prec,
        'recall':           rec,
        'roc_auc':          auc,
        'true_positives':   int(tp),
        'false_positives':  int(fp),
        'true_negatives':   int(tn),
        'false_negatives':  int(fn),
    })

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(
            model.feature_importances_,
            index=X_test.columns
        ).sort_values(ascending=False)
        wandb.log({'feature_importance': wandb.Table(
            columns=['feature', 'importance'],
            data=[[f, round(float(v), 5)] for f, v in fi.items()]
        )})

    run.finish()

    print(f'\n✓  Run "{run_name}" logged to WandB')
    print(f'   Accuracy {acc:.1%}  |  F1 {f1:.3f}  |  ROC AUC {auc:.3f}  '
          f'|  FN {fn}  |  FP {fp}')


# ── 📉 Class distribution plot ────────────────────────────────────────────────

def plot_class_distribution(df, target='Attrition'):
    """
    Bar chart showing the class distribution in the dataset.
    """
    # Handle both string ('Yes'/'No') and numeric (1/0) encoded targets
    if set(df[target].dropna().unique()).issubset({'Yes', 'No'}):
        counts = (df[target] == 'Yes').sum(), (df[target] == 'No').sum()
    else:
        counts = (df[target] == 1).sum(), (df[target] == 0).sum()

    rate = counts[0] / len(df) if len(df) > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Left', 'Stayed'], counts,
                  color=[_RED, _BLUE], width=0.5, alpha=0.9)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Employees')
    ax.set_title(f'Class distribution: {rate:.1%} attrition rate',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    print(f'For every 1 employee who left, {round((1 - rate) / rate)} stayed.')
    print('This imbalance is the first problem to solve.')


# ── 🔧 Train clean model (protected features removed) ─────────────────────────

def train_clean_model(X_train, X_test, y_train, y_test,
                      features_to_remove, pos_weight, threshold=0.3):
    """
    Train an XGBoost model with specified features removed.
    Returns the trained model, the cleaned test set, and predictions.

    Parameters
    ----------
    X_train, X_test : DataFrames
    y_train, y_test : Series
    features_to_remove : list of column names to drop
    pos_weight        : scale_pos_weight value
    threshold         : decision threshold for predictions

    Returns
    -------
    model, X_test_clean, y_pred
    """
    import xgboost as xgb

    missing = [f for f in features_to_remove if f not in X_train.columns]
    if missing:
        print(f'Warning: these features were not found and will be skipped: {missing}')
    features_to_remove = [f for f in features_to_remove if f in X_train.columns]

    X_train_clean = X_train.drop(features_to_remove, axis=1)
    X_test_clean  = X_test.drop(features_to_remove, axis=1)

    model = xgb.XGBClassifier(
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )
    model.fit(X_train_clean, y_train)

    y_prob  = model.predict_proba(X_test_clean)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)

    print(f'Features removed: {features_to_remove}')
    print(f'Features remaining: {X_test_clean.shape[1]}')

    return model, X_test_clean, y_pred


# ── 📋 Model comparison table ─────────────────────────────────────────────────

def compare_models(y_test, models, preds):
    """
    Print a side-by-side comparison of accuracy, F1, and ROC AUC
    for two or more models.

    Parameters
    ----------
    y_test  : true labels
    models  : dict of {name: (model, X_test)} for ROC AUC calculation
    preds   : dict of {name: y_pred} for accuracy and F1
    """
    names = list(preds.keys())
    header = f"{'':20}" + ''.join(f"{n:>12}" for n in names)
    print(header)
    print('-' * (20 + 12 * len(names)))

    accs = [accuracy_score(y_test, preds[n]) for n in names]
    f1s  = [f1_score(y_test, preds[n], pos_label=1, zero_division=0) for n in names]
    aucs = [roc_auc_score(y_test, models[n][0].predict_proba(models[n][1])[:,1])
            for n in names]

    print(f"{'Accuracy':20}" + ''.join(f"{v:>12.1%}" for v in accs))
    print(f"{'F1 (attrition)':20}" + ''.join(f"{v:>12.3f}" for v in f1s))
    print(f"{'ROC AUC':20}" + ''.join(f"{v:>12.3f}" for v in aucs))

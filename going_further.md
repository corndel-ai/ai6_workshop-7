# Going Further: PR AUC

**Workshop 7: Production Grade Classification**

---

## Why this matters for what you just built

In the workshop you used ROC AUC to evaluate your models. You saw values in the 0.6s and had to decide whether that was good enough for deployment. The ROC AUC told you something useful, but it was not telling you the whole story.

This document introduces PR AUC: a different way of measuring the same models that is more honest about what they can actually do when the positive class is rare.

Your attrition dataset has 85% stayed and 15% left. That imbalance is exactly the situation where ROC AUC flatters models and PR AUC does not.

---

## The sandwich inspector problem

Imagine you are a quality inspector at a sandwich factory. You check 1,000 sandwiches a day. 850 of them are fine. 150 have the wrong filling.

Your manager asks you to evaluate two inspection systems.

**System A** flags every single sandwich as faulty. It catches all 150 wrong sandwiches. But it also stops 850 perfectly good ones from leaving the factory.

**System B** passes every single sandwich without checking. It catches none of the 150 wrong ones. But it never causes a production halt.

If you measured both systems with accuracy, System B scores 85%. System A scores 15%. System B looks like the better system.

This is the accuracy trap. You already know this from the workshop.

Now ask: what does ROC AUC say about System B?

ROC AUC plots the true positive rate (how many wrong sandwiches you caught) against the false positive rate (how many good ones you wrongly stopped). System B catches no wrong sandwiches, so its true positive rate is 0. Its false positive rate is also 0, because it never stops anything. On the ROC curve it sits at the origin: mediocre, but not catastrophically low.

A model that predicts the majority class for everything still achieves a ROC AUC of around 0.5 on an imbalanced dataset. Not great, but not obviously broken either. It looks like it has learned something.

PR AUC is harsher. System B has a recall of zero, and its precision is undefined because it never flags anything. Scikit-learn scores a model that gives every case the same score as the proportion of positive cases, here about 0.15. That is the floor for PR AUC, and it makes a useless model look useless. The floor for ROC AUC is 0.5, which looks middling.

---

## What PR AUC actually measures

The precision-recall curve plots two things at every possible threshold:

- **Precision:** of all the employees (sandwiches, patients, transactions) you flagged, what proportion were genuinely positive?
- **Recall:** of all the genuinely positive cases, what proportion did you catch?

As you lower the threshold, recall goes up (you catch more real cases) but precision goes down (you also flag more false ones). As you raise the threshold, the opposite happens.

The curve shows this tradeoff across every threshold at once. PR AUC is the area under that curve: a single number summarising how well the model balances precision and recall across all possible operating points.

A perfect model has a PR AUC of 1.0. A model that predicts randomly has a PR AUC equal to the proportion of positive cases in the dataset: the proportion of positive cases: 0.146 across the whole attrition dataset, and about 0.15 in the validation set (22 leavers out of 147). That is the baseline. A score below it is worse than random.

This is the key difference from ROC AUC, where a random model scores 0.5 regardless of class balance. With PR AUC, the baseline shifts with the data. A PR AUC of 0.35 on a dataset with 15% positives is a clear improvement on the 0.15 baseline. The same score on a balanced dataset would be weak.

---

## The hunter-gatherer framing

Imagine a small tribe of 100 people who eat by foraging. Each day they decide which patches of forest to search. 15 patches contain food. The rest contain nothing.

A good forager remembers which patches had food last season (recall) but does not waste everyone's energy on patches that turned out to be empty (precision). The tribe cannot afford to search everywhere, and going hungry because you only visited empty patches is equally costly.

ROC AUC measures whether the forager ranks good patches above bad ones in general. That is useful but forgiving: as long as the overall ranking is decent, the score stays reasonable even if most food patches are missed.

PR AUC measures whether the patches you actually visit tend to have food, and whether you are visiting most of the food patches. Both matter, neither gets to hide behind the other.

For attrition prediction: your HR team cannot afford to have a retention conversation with every employee (precision matters). But they also cannot afford to miss most of the people genuinely at risk of leaving (recall matters). PR AUC measures both at once.

---

## How it is calculated

To generate the PR curve for your attrition model:

1. Run `predict_proba()` on your validation set to get a probability for each employee.
2. Sort those probabilities from highest to lowest.
3. At each probability value as a threshold, calculate precision and recall for that threshold.
4. Plot recall on the x-axis and precision on the y-axis.
5. Calculate the area under that curve.

In practice, step 5 has a subtlety. The trapezoidal rule joins adjacent points with straight lines and sums the areas. That is not a valid way to interpolate a precision-recall curve, because precision does not change in a straight line between thresholds. The two methods give different values: in our run the trapezoidal figure was slightly lower than the average precision (0.247 against 0.265). Scikit-learn's primary PR AUC function avoids the problem.

```python
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

y_prob = model_clean.predict_proba(X_val_clean)[:, 1]

# Preferred: average_precision_score uses a weighted mean of precision
# at each threshold, weighted by the change in recall
ap = average_precision_score(y_val, y_prob)
print(f'Average Precision (PR AUC): {ap:.3f}')

# Alternative: compute the curve first, then calculate AUC
precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
pr_auc = auc(recall, precision)
print(f'PR AUC (trapezoidal): {pr_auc:.3f}')

# These will give slightly different values.
# average_precision_score is the standard recommendation.
```

`average_precision_score` is the function to use. It computes a weighted mean of precision values at each threshold, weighted by the change in recall from the previous threshold. This avoids interpolating between points.

---

## Running it on your model

Open your notebook and add a cell after the cell in Part 6 that trains the clean model. You already have `y_prob`, `y_val`, `X_val_clean`, and `model_clean` in memory.

```python
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# PR AUC for your clean model
y_prob_clean = model_clean.predict_proba(X_val_clean)[:, 1]
ap = average_precision_score(y_val, y_prob_clean)

# Plot the curve
precision, recall, _ = precision_recall_curve(y_val, y_prob_clean)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision, color='steelblue', lw=2, label=f'PR AUC = {ap:.3f}')
ax.axhline(y=y_val.mean(), color='grey', linestyle='--',
           label=f'Baseline (random) = {y_val.mean():.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve: Clean Model')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

print(f'PR AUC:  {ap:.3f}')
print(f'ROC AUC: {roc_auc_score(y_val, y_prob_clean):.3f}')
print(f'Baseline PR AUC (random model): {y_val.mean():.3f}')
```

Compare the two numbers. In our run of the solved notebook the clean model scored a ROC AUC of 0.69 and a PR AUC of 0.27 on the validation set (0.29 with older library versions). Your figures will differ a little, because library versions and your own feature decisions change them. The rest of this section uses 0.69 and 0.27.

The second number looks worse. But ask what a useless model would score on each scale.

A model that predicts the majority class every time (stayed, stayed, stayed, for every employee) scores around 0.5 on ROC AUC. Not great, but not obviously broken. It looks like it has learned something.

Your model scored 0.27 on the PR scale. A useless model would score about 0.15 on the same scale. To understand why the useless model lands at exactly 0.146, imagine setting your threshold very low, so the model flags every employee as at risk. Recall is 1.0: you caught everyone who left. But precision is just 0.15: only about 15% of the people you flagged actually left, because that is all there were. Now raise the threshold so high that the model flags nobody. Recall drops to 0. Precision is undefined. Everywhere in between, precision averages out to roughly the proportion of positives in the dataset, because that is the rate at which you encounter genuine cases no matter where you draw the line. 

In other words: the PR AUC of a useless model is always equal to the class prevalence. For your dataset, that is about 0.15. The baseline is built into the data itself, not into an arbitrary convention like 0.5. This makes PR AUC unusually honest. You do not need to run an experiment to know what random performance looks like. You just look at how rare the positive class is.

And now you can see that the "relative" floor is different. ROC AUC runs from 0.5 to 1.0. PR AUC runs from about 0.15 to 1.0. The scales are not the same length. So, look at where your model sits on each one:
On the ROC scale, your model scores 0.69. The distance from floor to perfect is 0.5. Your model has covered 0.19 of that distance: about 38% of the way.
On the PR scale, your model scores 0.27. The distance from floor to perfect is 0.85. Your model has covered 0.12 of that distance: about 14% of the way.

The improvement over random is real: roughly 1.4 times the floor on the ROC scale (0.69 against 0.5) and 1.8 times on the PR scale (0.27 against 0.15). **But the PR scale shows clearly that most of the distance between a useless classifier and a genuinely useful one has not been covered yet.** ROC AUC at 0.69 did not show you that, because it absorbs the majority class into the calculation and smooths over the gap.

---

## Limitations to keep in mind

**The baseline varies by dataset.** You now know that the PR AUC baseline for a useless model is always equal to the class prevalence. On your attrition data that is about 0.15, so a PR AUC of 0.27 is meaningful. On a fraud dataset with 0.1% positive cases, the baseline would be 0.001, and a PR AUC of 0.40 would be extraordinary. On a balanced dataset with 50% positives, the baseline is 0.5, and PR AUC and ROC AUC become more comparable. The number means nothing without knowing the prevalence, so always state both together.

**Sensitive to class distribution.** If your attrition rate changes over time (a recession, a new pay policy, a wave of resignations), the PR AUC of the same model will change even if the model itself has not changed. This is not a bug: it is the metric correctly reflecting that the problem has become harder or easier. But it means you cannot compare PR AUC values across different time periods without accounting for prevalence.

**Harder to explain to stakeholders.** ROC AUC has a clean probabilistic interpretation: the probability that the model ranks a random positive above a random negative. PR AUC does not have an equally clean story. For a compliance reviewer or HR director, you will probably still reach for the confusion matrix and the threshold table. PR AUC is a tool for you, the engineer, not a number for the board deck.

---

## When to use which

| Situation | Use |
|:--|:--|
| Balanced dataset, general model comparison | ROC AUC |
| Imbalanced dataset, minority class matters | PR AUC |
| Communicating to non-technical stakeholders | Confusion matrix at chosen threshold |
| Comparing models at a specific operating point | F1, precision, recall at that threshold |
| Selecting between models with different thresholds | PR AUC or ROC AUC depending on balance |

For attrition prediction: use both ROC AUC and PR AUC during development. Present the confusion matrix and F1 at your chosen threshold in the model card.

---

## Further reading

- Deepchecks glossary: [PR AUC](https://deepchecks.com/glossary/pr-auc/): clear and concise, the source this document builds on
- Scikit-learn documentation: [sklearn.metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- Davis and Goadrich (2006): *The Relationship Between Precision-Recall and ROC Curves*: the original paper establishing why PR curves are more informative for imbalanced problems, can be found [HERE](https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf)(accessed April 2026).

---

## Scikit-learn Pipelines and ColumnTransformer

The notebook encodes categorical features using `.cat.codes` before the split. This is a deliberate simplification so you can see what encoding does step by step.

In a production workflow you would wrap preprocessing and modelling into a single scikit-learn Pipeline. This has two advantages: it prevents data leakage (the encoder is fitted on training data only) and it makes the workflow reproducible as a single object.

Here is what the notebook's encoding and model training would look like as a Pipeline. It starts from the raw file, because the notebook's `X_train` has already been encoded:

```python
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

raw = pd.read_csv('ibm_attrition.csv')
y_raw = (raw['Attrition'] == 'Yes').astype(int)
X_raw = raw.drop(columns=['Attrition', 'EmployeeCount', 'EmployeeNumber',
                          'Over18', 'StandardHours'], errors='ignore')
X_tr, X_va, y_tr, y_va = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

# Identify column types (before any encoding)
cat_cols = X_tr.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols = [c for c in X_tr.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols),
])

pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', xgb.XGBClassifier(
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )),
])

pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_va)
```

The key difference: `preprocessor.fit()` happens inside `pipe.fit()`, which only sees the training set. When you call `pipe.predict(X_va)`, the same transformation is applied without re-fitting. That is the correct pattern.

For more detail, see:

- [Scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)

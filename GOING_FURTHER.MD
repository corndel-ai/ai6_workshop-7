# Going Further: PR AUC

**Workshop 7: Production Grade Classification**

---

## Why this matters for what you just built

In the workshop you used ROC AUC to evaluate your models. You saw values around 0.63 to 0.70 and had to decide whether that was good enough for deployment. The ROC AUC told you something useful, but it was not telling you the whole story.

This document introduces PR AUC: a different way of measuring the same models that is more honest about what they can actually do when the positive class is rare.

Your attrition dataset has 85% stayed and 15% left. That imbalance is exactly the situation where ROC AUC flatters models and PR AUC does not.

---

## The subway sandwich problem

Imagine you are a quality inspector at a sandwich factory. You check 1,000 sandwiches a day. 850 of them are fine. 150 have the wrong filling.

Your manager asks you to evaluate two inspection systems.

**System A** flags every single sandwich as faulty. It catches all 150 wrong sandwiches. But it also stops 850 perfectly good ones from leaving the factory.

**System B** passes every single sandwich without checking. It catches none of the 150 wrong ones. But it never causes a production halt.

If you measured both systems with accuracy, System B scores 85%. System A scores 15%. System B looks like the better system.

This is the accuracy trap. You already know this from the workshop.

Now ask: what does ROC AUC say about System B?

ROC AUC plots the true positive rate (how many wrong sandwiches you caught) against the false positive rate (how many good ones you wrongly stopped). System B catches no wrong sandwiches, so its true positive rate is 0. Its false positive rate is also 0, because it never stops anything. On the ROC curve it sits at the origin: mediocre, but not catastrophically low.

A model that predicts the majority class for everything still achieves a ROC AUC of around 0.5 on an imbalanced dataset. Not great, but not obviously broken either. It looks like it has learned something.

PR AUC would give System B a score of 0. If you never catch any wrong sandwiches, your precision is undefined and your recall is zero. The metric collapses immediately. PR AUC has no patience for a model that ignores the minority class.

---

## What PR AUC actually measures

The precision-recall curve plots two things at every possible threshold:

- **Precision:** of all the employees (sandwiches, patients, transactions) you flagged, what proportion were genuinely positive?
- **Recall:** of all the genuinely positive cases, what proportion did you catch?

As you lower the threshold, recall goes up (you catch more real cases) but precision goes down (you also flag more false ones). As you raise the threshold, the opposite happens.

The curve shows this tradeoff across every threshold at once. PR AUC is the area under that curve: a single number summarising how well the model balances precision and recall across all possible operating points.

A perfect model has a PR AUC of 1.0. A model that predicts randomly has a PR AUC equal to the proportion of positive cases in the dataset: approximately 15% for your attrition data (well, actually: 0.146). That is the baseline. To be precise, anything below 0.146 is *worse than random*.

This is the key difference from ROC AUC, where a random model scores 0.5 regardless of class balance. With PR AUC, the baseline shifts with the data. A PR AUC of 0.35 on a dataset with 15% positives is actually quite good. The same score on a balanced dataset would be weak.

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

In practice, step 5 has a subtlety. The trapezoidal rule (connecting adjacent points with straight lines and summing the areas) underestimates the true area because the precision-recall curve is not linear between points. Scikit-learn uses a different approach for its primary PR AUC function that handles this more accurately.

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

`average_precision_score` is the function to use. It computes a weighted mean of precision values at each threshold, weighted by the change in recall from the previous threshold. This avoids the underestimation problem of the trapezoidal rule.

---

## Running it on your model

Open your notebook and add a cell after Section 6. You already have `y_prob`, `y_val`, `X_val_clean`, and `model_clean` in memory.

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

Compare the two numbers. ROC AUC: 0.702. PR AUC: 0.275.

The second number looks worse. But ask what a useless model would score on each scale.

A model that predicts the majority class every time (stayed, stayed, stayed, for every employee) scores around 0.5 on ROC AUC. Not great, but not obviously broken. It looks like it has learned something.

Your model scored 0.275 on the PR scale. A useless model would score 0.146 on the same scale. To understand why the useless model lands at exactly 0.146, imagine setting your threshold very low, so the model flags every employee as at risk. Recall is 1.0: you caught everyone who left. But precision is just 0.146: only 14.6% of the people you flagged actually left, because that is all there were. Now raise the threshold so high that the model flags nobody. Recall drops to 0. Precision is undefined. Everywhere in between, precision averages out to roughly the proportion of positives in the dataset, because that is the rate at which you encounter genuine cases no matter where you draw the line. 

In other words: the PR AUC of a useless model is always equal to the class prevalence. For your dataset, that is 0.146. The baseline is built into the data itself, not into an arbitrary convention like 0.5. This makes PR AUC unusually honest. You do not need to run an experiment to know what random performance looks like. You just look at how rare the positive class is.

And now you can see that the "relative" floor is different. ROC AUC runs from 0.5 to 1.0. PR AUC runs from 0.146 to 1.0. The scales are not the same length. So, look at where your model sits on each one:
On the ROC scale, your model scores 0.702. The distance from floor to perfect is 0.5. Your model has covered 0.202 of that distance: about 40% of the way.
On the PR scale, your model scores 0.275. The distance from floor to perfect is 0.854. Your model has covered 0.129 of that distance: about 15% of the way.

The improvement over random is real, roughly 1.9 times on both scales. **But the PR scale shows clearly that most of the distance between a useless classifier and a genuinely useful one has not been covered yet.** ROC AUC at 0.702 did not show you that, because it was absorbing the majority class into the calculation and smoothing over the gap!

---

## Limitations to keep in mind

**The baseline varies by dataset.** You now know that the PR AUC baseline for a useless model is always equal to the class prevalence. On your attrition data that is 0.146, so a PR AUC of 0.275 is meaningful. On a fraud dataset with 0.1% positive cases, the baseline would be 0.001, and a PR AUC of 0.40 would be extraordinary. On a balanced dataset with 50% positives, the baseline is 0.5, and PR AUC and ROC AUC become more comparable. The number means nothing without knowing the prevalence, so always state both together.

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

- Deepchecks glossary: [PR AUC](https://deepchecks.com/glossary/pr-auc/) : clear and concise, the source this document builds on
- Scikit-learn documentation: [sklearn.metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- Davis and Goadrich (2006): *The Relationship Between Precision-Recall and ROC Curves* : the original paper establishing why PR curves are more informative for imbalanced problems, can be found [HERE](https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf)(accessed April 2026).

---

## Scikit-learn Pipelines and ColumnTransformer

The notebook encodes categorical features using `.cat.codes` before the split. This is a deliberate simplification so you can see what encoding does step by step.

In a production workflow you would wrap preprocessing and modelling into a single scikit-learn Pipeline. This has two advantages: it prevents data leakage (the encoder is fitted on training data only) and it makes the workflow reproducible as a single object.

Here is what the notebook's encoding and model training would look like as a Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

# Identify column types (before any encoding)
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
num_cols = X_train.select_dtypes(exclude='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols),
])

pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', xgb.XGBClassifier(
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_val)
```

The key difference: `preprocessor.fit()` happens inside `pipe.fit()`, which only sees `X_train`. When you call `pipe.predict(X_val)`, the same transformation is applied without re-fitting. That is the correct pattern.

For more detail, see:

- [Scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)

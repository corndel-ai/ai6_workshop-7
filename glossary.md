# Glossary
## Workshop 7: Production Grade Classification

---

### SHAP (SHapley Additive exPlanations)

A method for explaining what a machine learning model is doing. It assigns each feature a contribution score for each individual prediction, showing how much that feature pushed the prediction toward or away from the positive class.

The name comes from game theory. If several players (features) cooperate to produce an outcome (a prediction), how do you fairly divide the credit between them? SHAP uses Shapley values to answer that question.

In practice: a positive SHAP value for a feature means it pushed this prediction toward attrition. A negative value means it pushed away from it. The size tells you how much.

**Why it matters here:** XGBoost can tell you which features it uses most overall. SHAP tells you why it made a specific prediction for a specific employee.

---

### Class imbalance

When one class in your dataset is much more common than the other. In this dataset, around 85% of employees stayed and 15% left. A model trained without accounting for this will learn that predicting "stayed" is usually correct and largely stop trying to identify the minority class.

**Example:** If your training set has 1,000 examples of "stayed" and 150 of "left", the model quickly learns that guessing "stayed" every time gives it 87% accuracy. It has learned nothing useful about who actually leaves.

---

### scale_pos_weight

An XGBoost parameter that tells the model to treat each positive case (attrition) as if it were worth more than each negative case (stayed). Set it to the ratio of negatives to positives and the model is forced to pay attention to the minority class.

It is the simplest fix for class imbalance: one parameter, no resampling, no new data. If you have 1,000 stayed and 150 left, set it to 1000/150 = 6.7. Each attrition case now counts as 6.7 stayed cases in the loss calculation.

---

### SMOTE (Synthetic Minority Oversampling Technique)

A resampling method that creates synthetic examples of the minority class rather than simply duplicating existing ones. It finds real minority examples that are close to each other in feature space and generates new points between them.

More sophisticated than `scale_pos_weight` but also more complex to reason about. For this dataset, `scale_pos_weight` is sufficient and more transparent.

**If you want to go further:** try `from imblearn.over_sampling import SMOTE` and compare your results to the `scale_pos_weight` approach.

---

### Decision threshold

The probability cutoff above which a model flags a prediction as positive. At 0.5, any employee the model believes has more than a 50% chance of leaving gets flagged. Raising it makes the model more conservative: fewer flags, fewer false alarms, but more missed cases. Lowering it does the opposite.

It is not a fixed property of the model. It is a design decision with consequences for real people, and it should be set deliberately, not left at the default.

---

### ROC AUC (Receiver Operating Characteristic: Area Under the Curve)

A single number summarising how well a classifier separates the two classes across all possible thresholds. It measures the probability that the model ranks a randomly chosen positive case above a randomly chosen negative case.

0.5 means random. 1.0 means perfect. In practice, anything above 0.7 is considered moderate for most real-world problems. What counts as good depends on the domain and on what you are comparing against.

_Fun fact: The Receiver Operating Characteristic (ROC) curve is named for its origin in World War II, where it was used to analyse how well radar receivers and their operators could distinguish enemy aircraft from background noise_

---

### Precision

Out of all the employees the model flagged as likely to leave, what proportion actually left? High precision means few false alarms. A precision of 0.4 means 4 in every 10 flags are genuine, and 6 are wasted investigations.

---

### Recall

Out of all the employees who actually left, what proportion did the model catch? High recall means few missed cases. A recall of 0.8 means the model caught 8 in every 10 people who left, and missed 2.

---

### F1 score

The harmonic mean of precision and recall, giving a single number that balances both. Useful when the class distribution is unequal, and when you cannot afford to sacrifice one metric entirely for the other. A model that gets precision or recall to zero will score F1 of zero, even if the other metric is perfect. (The harmonic mean is a type of average that punishes extreme imbalances: two numbers that are far apart will always produce a lower harmonic mean than two numbers that are close together, even if their ordinary average is the same.)

---

### Equality Act 2010 (UK)

UK legislation that makes it unlawful to discriminate against people on the basis of protected characteristics in employment, education, and services. The nine protected characteristics are: age, disability, gender reassignment, marriage and civil partnership, pregnancy and maternity, race, religion or belief, sex, and sexual orientation.

Using a protected characteristic as a feature in a model that influences employment decisions creates legal exposure, even if discrimination was not intended. The model does not need to be biased in an obvious way. If it uses age as a predictor and age correlates with another protected characteristic in the data, the effect can still be discriminatory.

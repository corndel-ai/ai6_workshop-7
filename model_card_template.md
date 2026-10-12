# Model Card: HR Attrition Classifier

*Complete this card before deploying the model. It is the document you would share with a legal or compliance reviewer, a senior stakeholder, or a new team member inheriting the system.*

---

## Model details

| | |
|:--|:--|
| **Model type** | XGBoost binary classifier |
| **Version** | |
| **Date trained** | |
| **Trained by** | |
| **Contact** | |

---

## Intended use

**What the model is for:**


**What the model is not for:**


**Who will use the outputs:**


---

## Data

| | |
|:--|:--|
| **Dataset** | IBM HR Analytics (fictional teaching dataset), 1,470 employees |
| **Training set** | 1,176 rows |
| **Validation set** | 147 rows |
| **Held-out test set** | 147 rows |
| **Target** | Attrition (Yes/No) |
| **Attrition rate** | About 15% |

**Features removed and why:**

| Feature | Reason for removal |
|:--|:--|
| | |

---

## Performance

Report the validation figures from the notebook, then the held-out figures from Part 10.

| Metric | Value |
|:--|:--|
| ROC AUC | |
| F1 (attrition class) | |
| Precision | |
| Recall | |
| Missed (FN) | |
| False alarms (FP) | |
| Decision threshold | |

**Why this threshold was chosen:**


---

## Known limitations

*What does the model not do well? Where does it fail?*


---

## Risks and mitigations

| Risk | Mitigation |
|:--|:--|
| Indirect discrimination via correlated features | |
| Model drift over time | |
| Missed departures (false negatives) | |
| Unnecessary interventions (false alarms) | |

---

## Ethical considerations

**Who bears the cost of a missed departure in this system:**


**What recourse do they have:**


**Was a DPIA conducted:**


---

## Oversight and accountability

| Role | Responsibility |
|:--|:--|
| Deployment decision owner | |
| Post-deployment monitoring owner | |
| Withdrawal criteria | |

---

*This model card follows the structure proposed by Mitchell et al. (2019). See glossary.md for definitions of terms used here.*

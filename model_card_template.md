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
| **Dataset** | IBM HR Analytics (synthetic), 1,470 employees |
| **Training set** | 1,176 rows |
| **Test set** | 294 rows |
| **Target** | Attrition (Yes/No) |
| **Attrition rate** | ~15% |

**Features removed and why:**

| Feature | Reason for removal |
|:--|:--|
| Age | Protected characteristic (Equality Act 2010) |
| Gender | Protected characteristic (Equality Act 2010) |
| MaritalStatus | Protected characteristic (Equality Act 2010) |
| | |

---

## Performance

| Metric | Value |
|:--|:--|
| ROC AUC | |
| F1 (attrition class) | |
| Precision | |
| Recall | |
| False negatives | |
| False positives | |
| Decision threshold | |

**Why this threshold was chosen:**


---

## Known limitations

*What does the model not do well? Where does it fail?*


---

## Risks and mitigations

| Risk | Mitigation |
|:--|:--|
| Indirect discrimination via correlated features | Features reviewed against Equality Act; see feature removal above |
| Model drift over time | |
| False negatives (missed attrition) | |
| False positives (unnecessary interventions) | |

---

## Ethical considerations

**Who bears the false negative in this system:**


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

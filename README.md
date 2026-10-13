# Workshop 7: Production Grade Classification

**Corndel AI6 Level 6 ML Engineer Apprenticeship**
Unit 7: Supervised Machine Learning

---

## Start here

1. Open this repository in a Codespace (green **Code** button, then **Codespaces**, then **Create codespace**).
2. Open `7w_hr_attrition.ipynb` and work through it from the top, in order.
3. If you want a step-by-step walkthrough with screenshots, read `tutorial/7w_walkthrough.md` alongside the notebook.

You need no setup beyond the Codespace. A Weights & Biases (W&B) account is optional: the notebook records your runs to a local file if you do not have one. Pre-work instructions are in `7w_prework.md`.

---

## What this workshop is

A five-hour coach-facilitated session. You build, evaluate and defend a classifier that predicts which employees are at risk of leaving, using the IBM HR Analytics dataset. The model is deliberately imperfect. The session is about the decisions that make a model deployable, not only performant: choosing a metric that reflects the cost of failure, handling class imbalance, setting a threshold you can justify, and removing features that should not be in the model even when they help its score.

The use case is this. The HR director of a mid-sized technology company wants to know who is at risk of leaving. By the end of the session you can say which employees the model would flag, what each kind of error would cost the people involved, and what must be true before the model is used.

"Production grade" describes the classification model: rigorous evaluation, honest handling of imbalance, explainability, ethical feature decisions and documentation. The wider deployment pipeline (CI/CD, orchestration and monitoring) is outside this session.

---

## The route through the notebook

The notebook has eleven Parts in four phases. Each Part has a short explanation, a worked example and a task with a hint, a solution and a check cell.

| Phase | Part | What you do |
|:--|:--|:--|
| **Build** | 1. Meet and prepare the data | Load the data, encode the target, split into training, validation and held-out test sets |
| | 2. The accuracy trap | Train a default model and see why high accuracy can hide poor recall |
| | 3. Fix the class imbalance | Weight the minority class and compare the two models |
| | 4. Choose a threshold you can defend | Read the ROC curve and set a justified threshold |
| **Decide** | 5. See inside the model with SHAP | Find which features drive predictions |
| | 6. Decide which features stay | Apply a legal and ethical check and test your decision |
| **Ship** | 7. Track, compare and save your runs | Compare your runs, save and register the model |
| | 8. Serve the model | Read and run a FastAPI service built from your model |
| | 9. Agree the output and document the model | Choose an output format and complete the model card |
| **Defend** | 10. Stress-test and evaluate | Probe edge cases and run the final held-out evaluation |
| | 11. Hand over the model | Write a deployment gate and a three-slide summary (independent task) |

The notebook ends with a three-question reflection and a summary.

---

## Files in this repository

| File | Purpose |
|:--|:--|
| `7w_hr_attrition.ipynb` | The learner notebook. Work through it in order. |
| `utils.py` | Helper functions for plots, run tracking and success checks. Must sit next to the notebook. |
| `app.py` | The FastAPI service you run in Part 8. It reads the model and settings that the notebook saves. |
| `ibm_attrition.csv` | The dataset. Must sit next to the notebook. |
| `model_card_template.md` | The blank model card. The notebook copies it to `model_card.md` in Part 9. |
| `glossary.md` | Definitions for SHAP, DPIA, ROC AUC, the Equality Act and other terms. |
| `7w_prework.md` | Optional pre-work for learners: setting up a W&B account. Send before the session. |
| `going_further.md` | Extra reading and extension tasks for learners who finish early. |
| `tutorial/7w_walkthrough.md` | A screenshot walkthrough of the whole notebook. |
| `cheat_sheets/` | Two one-page visual summaries. |
| `coach/` | Coach material: `help.md` (troubleshooting) and `solutions/` (a completed notebook). |

Files the notebook creates while you work (`runs_log.csv`, `hr_attrition_model.joblib`, `model_meta.json`, `model_card.md`) are not committed unless you add them.

---

## KSB coverage

Primary KSBs are those most directly evidenced by work in this workshop. Secondary KSBs are reinforced here but were introduced earlier or are revisited later.

### Primary KSBs

| KSB | Description | Where evidenced |
|:--|:--|:--|
| **K7** | Performance metric selection in business context | Accuracy trap, recall, threshold justification (Parts 2 to 4) |
| **K19** | Problem-solving via test data analysis including edge-case data | Stress-testing with real validation profiles and counterfactual reasoning (Part 10) |
| **S5** | Create and deploy models | FastAPI service, joblib model saving (Parts 7 and 8) |
| **S6** | Document model lifecycle assets | Run history, model registration, model card, `app.py` (Parts 7, 8 and 9) |
| **S14** | Analyse test data, interpret results, evaluate solution suitability | Confusion matrix, ROC AUC, threshold table, stress-test profiles, final evaluation (Parts 2 to 4 and 10) |
| **S20** | Minimise algorithmic bias | SHAP analysis, feature removal, documented in the model card (Parts 5, 6 and 9) |
| **S28** | Produce and maintain technical documentation for the data product | Model card completed with real results (Part 9) |
| **B4** | Acts with integrity, legal and ethical requirements | Threshold justification, feature removal, accountability questions (Parts 4, 6, 9 and the Reflection) |
| **B5** | Operates in technical complexity and uncertainty | Borderline threshold decision, edge cases, held-out versus validation gap (Parts 4 and 10) |

### Secondary KSBs

| KSB | Description | Where evidenced |
|:--|:--|:--|
| K5 | ML methods and experiment tracking tools | XGBoost throughout; run tracking and model registration (Parts 2 to 7) |
| K29 | Own role in relation to organisational governance, legal and ethical practice | Model card, deployment gate, Reflection (Parts 9 and 11) |
| S11 | Regulatory, legal, ethical and governance issues at each stage | Feature selection, DPIA, Equality Act (Parts 5, 6 and 9) |
| S29 | Create reports and presentations for stakeholder approval and handover | Deployment gate and three-slide summary (Part 11) |

---

*Part of the Corndel AI6 curriculum. For curriculum queries contact the AI6 team.*

# Workshop 7: Production Grade Classification

**Corndel AI6 Level 6 ML Engineer Apprenticeship**
Unit 7: Supervised Machine Learning

---

## What this workshop is

A five-hour coach-facilitated session in which learners build, evaluate, and defend an HR attrition classifier using the IBM HR Analytics dataset. The model is deliberately imperfect. The session is structured around the decisions that make a model deployable, not just performant: choosing the right metric, handling class imbalance, setting a threshold you can justify, and removing features that should not be in the model even when they improve accuracy.

Every decision is logged to WandB. By the end of the session each learner has a dashboard showing four runs and the effect of each choice.

---

## Files in this repository

| File | Purpose |
|:--|:--|
| `7w_hr_attrition.ipynb` | The learner notebook. Work through this in order. |
| `utils.py` | Helper functions for plots and WandB logging. Must be in the same folder as the notebook. |
| `ibm_attrition.csv` | The dataset. Must be in the same folder as the notebook. |
| `glossary.md` | Definitions for SHAP, SMOTE, DPIA, ROC AUC, the Equality Act, and other terms used in the notebook. |
| `coach_playbook.md` | Timings, pedagogy intent, and troubleshooting for coaches. |
| `7w_prework.md` | Pre-work instructions for learners. Send this before the session. |
| `GOING_FURTHER.md` | Learners who finish early, encounter a break, or crave more reading, are encouraged to look here |

---

## Before the session

Learners need one thing: a WandB account and their API key. See `7w_prework.md` for the exact steps. Everything else runs in Codespaces.

Coaches should read `coach_playbook.md` before the session, particularly the troubleshooting section.

---

## Running the notebook

Open the repository in a Codespace. The notebook, utils.py, and ibm_attrition.csv must all be in the same directory. Run cells in order using Shift+Enter. The first cell checks for utils.py and raises a clear error if it is missing.

---

## KSB coverage

Primary KSBs are those most directly and substantially evidenced by work produced in this workshop. Secondary KSBs are reinforced here but were introduced in earlier units or will be revisited later.

### Primary KSBs

**These require the most focus during delivery and assessment.**

| KSB | Description | Where evidenced |
|:--|:--|:--|
| **K7** | Performance metric selection in business context | Accuracy trap, F1 vs recall, threshold justification (Sections 2-4) |
| **K19** | Problem-solving via test data analysis including edge-case data | Stress-testing with real test profiles and counterfactual reasoning (Section 12) |
| **S5** | Create and deploy models | FastAPI wrapper, joblib model saving (Sections 9-10) |
| **S6** | Document model lifecycle assets | WandB run history, model card, app.py (Sections 9, 11) |
| **S14** | Analyse test data, interpret results, evaluate solution suitability | Confusion matrix, ROC AUC, threshold table, stress-test profiles (Sections 2-4, 12) |
| **S20** | Minimise algorithmic bias | SHAP analysis, feature removal, documented in model card (Sections 5-6, 11) |
| **S28** | Produce and maintain technical documentation for the data product | Model card completed with real results (Section 11) |
| **B4** | Acts with integrity, legal and ethical requirements | Threshold justification, feature removal, accountability questions throughout |
| **B5** | Operates in technical complexity and uncertainty | Borderline threshold decision, stress-test edge cases, ROC AUC calibration |

### Secondary KSBs

Reinforcement of ideas introduced in earlier units or revisited in later ones.

| KSB | Description | Where evidenced |
|:--|:--|:--|
| K5 | ML methods and experiment tracking tools | XGBoost throughout; WandB logging and model registration (Sections 3-9) |
| K29 | Own role in relation to organisational governance, legal and ethical practice | Handoff discussion, accountability wrap-up, model card (Sections 10b, 13, 14) |
| S11 | Regulatory, legal, ethical and governance issues at each stage | Feature selection, DPIA, Equality Act, throughout Sections 5-14 |
| S29 | Create reports and presentations for stakeholder approval and handover | Three-slide executive summary, handoff discussion (Sections 13, 14) |

---

*Part of the Corndel AI6 curriculum. For curriculum queries contact the AI6 team.*

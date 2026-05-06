# Help
## Workshop 7: Production Grade Classification
### Corndel AI6 Level 6 ML Engineer Apprenticeship

---

## About the two notebook versions

The repo contains two notebook files.

**`7w_hr_attrition.ipynb`** is the learner version. All code cells are clean with no outputs. The four uncomment tasks have all three options commented out : learners must identify and uncomment the correct line themselves. There are no answers given away in comments.

**`solutions/7w_hr_attrition_solved.ipynb`** is the coach reference version. It contains the outputs from a full executed run, showing expected metric values, confusion matrices, WandB logging confirmations, stress-test probabilities, and risk tier counts. Use this to check what correct output looks like before the session, and to verify your own environment is working.

## Troubleshooting

**WandB login fails**
Comment out `wandb.login()` and uncomment `wandb.init(mode='disabled')`. Everything runs, no dashboard at the end. Learner creates account after the session.

**utils.py not found**
Learner opened the notebook directly. Close, go to the repo, open Codespace from there. File is present automatically.

**ibm_attrition.csv not found**
Same cause as above.

**Uncomment task produces an error**
Expected for the wrong options. Ask which line they uncommented and why. Use the error as a teaching moment.

**scale_pos_weight inverted**
Common mistake. Ratio of stayed to left, not left to stayed. Ask: if there are 6 stayed for every 1 left, how much extra weight should each left case get?

**Learner picks threshold without justifying it**
Ask: what does a false negative mean for an employee in this system? What does a false alarm mean? Which would you rather explain to the HR director?

**Learner wants to keep Age in the model**
Do not override them. Ask them to document the reasoning in the code comment and make sure they can answer Q3 in the wrap-up.

**Model card feels overwhelming**
Performance table first, using WandB numbers. The rest follows.

**joblib or fastapi not installed**
Run: `pip install joblib fastapi uvicorn pydantic --break-system-packages`

**app.py column mismatch**
Learner changed FEATURES_TO_REMOVE and ran app.py without regenerating. Ask them to rerun the app.py writer cell.

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

## GOING FURTHER

| `GOING_FURTHER.md` | Learners who finish early, encounter a break, or crave more reading, are encouraged to look here |

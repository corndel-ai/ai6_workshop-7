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

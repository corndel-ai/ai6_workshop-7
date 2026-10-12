# Coach help
## Workshop 7: Production Grade Classification

---

## About the notebook versions

**`7w_hr_attrition.ipynb`** (repository root) is the learner version. Every task has starter code, a collapsible hint, a collapsible solution and a check cell that prints whether the answer is right. Check cells never stop the notebook, so a learner who is stuck can keep going.

**`coach/solutions/7w_hr_attrition_solved.ipynb`** is the coach reference version, with every task completed and the outputs from a full run. Use it to see what correct output looks like before the session. Copy it next to `utils.py` and `ibm_attrition.csv` before running it, because the notebook looks for those files in its own folder.

The solved notebook does not complete the model card, because that document is written by the learner. Its model card check therefore reports "Not yet".

Exact metric values depend on the library versions installed in the Codespace, so a learner's numbers can differ from the solved notebook. The notebook text that interprets results is computed from the learner's own numbers, so small differences are expected.

## Troubleshooting

**W&B login fails or the learner has no account**
Set `TRACKING_MODE = 'local'` in the tracking cell. Every run is still saved to `runs_log.csv` and all later cells work. The same fallback applies automatically if the learner presses Escape at the login prompt.

**The W&B login prompt does not appear in the notebook**
It appears in the search bar at the top of the Codespaces window. The image `wandb_login_location.png` in the notebook shows where.

**utils.py not found**
The learner opened the notebook outside the repository. Close it, open the Codespace from the repository, and reopen the notebook.

**ibm_attrition.csv not found**
Same cause as above.

**A success check says "Not yet"**
Read the message with the learner. Each message names what to look for. Ask which line they wrote and why.

**scale_pos_weight inverted**
Common mistake. The ratio is stayed to left, not left to stayed. Ask: if there are 6 stayed for every 1 left, how much extra weight should each left case get?

**Learner picks a threshold without justifying it**
Ask: what does a missed departure mean for an employee in this system? What does a false alarm mean? Which would you rather explain to the HR director?

**Learner wants to keep Age in the model**
Do not override them. Ask them to write the reasoning in `FEATURE_REASON` and make sure they can answer Reflection question 2.

**Model card feels overwhelming**
Performance table first, using the numbers printed by the notebook. The rest follows from the reasoning already written in `THRESHOLD_REASON` and `FEATURE_REASON`.

**joblib, fastapi or uvicorn not installed**
Run `pip install -r requirements.txt` in a Codespace terminal.

**The API reports a missing field or a mismatch**
The learner changed `FEATURES_TO_REMOVE` after saving the settings. Ask them to re-run the cells from Part 6, then the `save_model_meta` cell in Part 8, and restart the API.

**Opening the API address shows "Not Found"**
Expected. The API defines `/health`, `/predict` and `/docs`, not a root address.

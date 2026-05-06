# Coach Playbook
## Workshop 7: Production Grade Classification
### Corndel AI6 Level 6 ML Engineer Apprenticeship

---

## About the two notebook versions

The repo contains two notebook files.

**`7w_hr_attrition.ipynb`** is the learner version. All code cells are clean with no outputs. The four uncomment tasks have all three options commented out : learners must identify and uncomment the correct line themselves. There are no answers given away in comments.

**`7w_hr_attrition_COACH.ipynb`** is the coach reference version. It contains the outputs from a full executed run, showing expected metric values, confusion matrices, WandB logging confirmations, stress-test probabilities, and risk tier counts. Use this to check what correct output looks like before the session, and to verify your own environment is working.

Do not share the coach version with learners.

---

## Delivery notes

This workshop requires no slides. Do not prepare a deck. Everything the learner needs is in the notebook and the supporting files in the repo.

**Safeguarding** must be covered in the first 15 minutes but not from a slide. Cover it conversationally: name the reporting contact for your organisation, confirm learners know how to raise a concern during or after the session, and check in on any access or support needs. This takes five minutes if done directly and twice as long if done from a list.

**Screen sharing** is the primary tool for the session. Share your screen when running the notebook so learners can see what a clean run looks like before they start their own. This is particularly useful at the setup cell and at the WandB login prompt.

**Breakout rooms:** move between breakout rooms regularly, especially during sections 4 (threshold decision), 6 (feature selection), and 11 (model card). Learners who are stuck rarely ask for help in the main room. A brief visit to each breakout room every 15-20 minutes catches this early.

**Polls** are useful at two moments: after Section 2 ("which model would you choose right now?") to surface prior instincts before the threshold discussion, and after Section 6 ("did you remove Age?") to show the group how different their decisions were. Neither poll needs preparation : both can be launched ad hoc.

**Whiteboards** work well for Section 13 (pre-deployment checklist) if the group is remote. Ask each learner to add one item to a shared board rather than filling in the comment block individually. This makes the discussion more visible and surfaces disagreements naturally.

Use these tools where they add something. Do not use them for the sake of it.

---

## Pedagogy intent and programme positioning

Workshop 7 is the practical capstone of Unit 7. Learners have covered evaluation metrics (7.2), advanced modelling (7.3), and the deployment decision framework (7.4). This session asks them to apply all three to a real dataset, with a model that is deliberately imperfect, and with decisions they have to justify rather than just make.

The central question is the gap between a model that performs on a holdout set and one you would trust to influence employment decisions. That gap is where K7, S11, S14, S20, B4 and B5 live.

The morning builds the problem. The afternoon solves it, deploys it, and asks what comes next. The productive discomfort is intentional: there is no single correct threshold, no correct feature list, no correct output format. Learners who want a right answer need to be redirected to justifying a defensible one.

---

## Fixed session plan

**Total contact time: 5 hours (2.5h morning, 2.5h afternoon)**

### Morning | 09:00-11:30

| Time | Duration | Activity | What to watch for |
|:--|:--|:--|:--|
| 09:00 | 15 min | Safeguarding + warmup | Cover reporting procedures, name any support contacts, quick icebreaker connecting to the dataset topic |
| 09:15 | 15 min | Setup + WandB login | Anyone without API key: share yours temporarily, they create their own afterwards. utils.py error means they opened the notebook directly, not from the repo. |
| 09:30 | 10 min | Section 1: The data | Learners should notice the 15% attrition rate before you name it. Ask: what do you notice? |
| 09:40 | 25 min | Section 2: Naive model | The penny-drop moment at the confusion matrix. Give it time. Do not rush to the fix. A learner who says "83% is fine" needs a prompt: how many people who actually left did the model catch? |
| 10:05 | 20 min | Section 3: Fix the imbalance | The uncomment task surfaces whether the ratio direction is understood. Wrong answer: good teaching moment. |
| 10:25 | 30 min | Section 4: Choose a threshold | Most important conversation of the morning. Push back on anyone who picks a number without naming who bears the consequence of each error type. "It gives a better F1" is not a justification. |
| 10:55 | 20 min | Section 5: SHAP | Ask: which features surprised you? Which ones worry you? Do not name Age yourself, let learners find it. |
| 11:15 | 15 min | Section 6: Feature selection | Flag the DPIA collapsible. The updated table with Possibly column usually generates discussion. The point is not to reach the right feature list but to document the reasoning. |
| **11:30** | | **End of morning** | |

### Afternoon | 13:00-15:30

| Time | Duration | Activity | What to watch for |
|:--|:--|:--|:--|
| 13:00 | 15 min | Section 7: WandB dashboard | Ask each learner to name the single change that made the biggest difference to false negatives. If someone cannot answer, they have not read the dashboard. |
| 13:15 | 20 min | Section 8: Wrap-up questions | Group discussion. Aim for the whole group to hear different answers to Q3 (feature removal justification). Q4 (withdrawal criteria) is often the most revealing. |
| 13:35 | 10 min | Section 9: Register the model | Fast. Confirm model is saved to disk before moving on. |
| 13:45 | 15 min | Section 10: FastAPI wrapper | They write app.py. Do not spend time on Kubernetes theory unless asked. The collapsible covers it if they want it. |
| 14:00 | 15 min | Section 10b: Agree the output | The risk tier calculation is the anchor. Ask: can your HR team action 47 conversations per cycle? If not, what changes? |
| 14:15 | 20 min | Section 11: Model card | Ask learners to fill in the performance table first using WandB numbers. That unlocks the rest. Bullet points in the limitations section are fine. |
| 14:35 | 15 min | Section 12: Stress-test | Three profiles, look at the results. The high-risk employee who did not actually leave is the most important row. Ask: what does that tell you? |
| 14:50 | 10 min | Section 13: Pre-deployment checklist | Quick comment block, then one person reads theirs to the group. |
| 15:00 | 10 min | Section 14: Present your work | Three-slide structure as a comment. Going further is optional and for after the session. |
| 15:10 | 5 min | Close | WandB project and model card are permanent evidence. Learning journal note. |
| **15:30** | | **End of session** | **15 min buffer for overruns** |

---

## Buffer management

The afternoon has 15 minutes of built-in buffer. Use it as follows, in order of priority:

1. Section 4 (threshold) almost always overruns. Give it the first 5 minutes of buffer if needed.
2. Section 8 (wrap-up questions) generates the most discussion. Give it up to 10 minutes if the group is engaged : this is K26 and B4 territory and the conversation is the evidence.
3. Sections 13 and 14 can be shortened to 5 minutes each if time is short. The comment blocks can be completed after the session.

---

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


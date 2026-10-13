# Welcome to Corndel AI6 Workshop 7: Production Grade Classification

## Pre-work: Optional Setup Before the Session

You already have a GitHub account from Workshop 6. That is all you need for Codespaces. **Nothing in this pre-work is required to take part.**

There is one optional step: create a Weights & Biases (W&B) account and get your API key ready. W&B gives you a dashboard that shows every model you train side by side. If you skip it, the notebook saves the same information to a local file, `runs_log.csv`, and you can compare your runs in the notebook itself.

---

## Option 1: Use W&B (about 10 minutes)

**Step 1.** Go to [wandb.ai](https://wandb.ai) and sign up. You can use your GitHub account to log in directly. You can alternatively make up most details during signup (name, organisation, job title) as long as you use a real email address. When prompted to choose something to try first (Weave, Models, etc.), pick any option. You need to choose one before you can access Settings, where your API key is.

**Step 2.** Once signed in, go to [wandb.ai/settings](https://wandb.ai/settings).

**Step 3.** Scroll to the **API keys** section and click **New key**. Give it any name.

**Step 4.** Copy the key immediately and paste it somewhere safe. You can only see the key once.

### On the day: where the W&B prompt appears

When you run the tracking cell, the prompt does not appear inside the notebook. It appears in the **search bar at the top of the Codespaces window**, the bar that normally shows the file name.

When it appears:

1. Select option **(2) Use an existing W&B account**.
2. Paste your API key when asked.

See `wandb_login_location.png` in the repo for a screenshot showing exactly where to look.

---

## Option 2: Skip W&B

If you do not want an account, or you ran out of time, do nothing now. On the day, in the tracking cell, change the setting to:

```python
TRACKING_MODE = 'local'
```

Your runs are saved to `runs_log.csv` and the rest of the notebook works in the same way. If you leave the setting as `'wandb'` and press Escape at the login prompt, or the login fails, the notebook also falls back to the local file and carries on.

---

If anything goes wrong, tell your coach.

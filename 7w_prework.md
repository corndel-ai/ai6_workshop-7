# Welcome to Corndel AI6 Workshop 7: Production Grade Classification

## Pre-work: One Thing to Do Before the Session

You already have a GitHub account from Workshop 6. That is all you need for Codespaces.

There is one new thing: create a WandB account and get your API key ready.

---

## Create a WandB account and get your API key

WandB tracks every model you train. By the end of the session you will have a dashboard showing all your runs side by side.

**Step 1.** Go to [wandb.ai](https://wandb.ai) and sign up. You can use your GitHub account to log in directly. You can alternatively make up most details during signup (name, organisation, job title) as long as you use a real email address. When prompted to choose something to try first (Weave, Models, etc.), pick any option. You need to choose one before you can access Settings, where your API key is.

**Step 2.** Once signed in, go to [wandb.ai/settings](https://wandb.ai/settings).

**Step 3.** Scroll to the **API keys** section and click **New key**. Give it any name.

**Step 4.** Copy the key immediately and paste it somewhere safe. **You can only see the key once.**

---

## On the day: where the WandB prompt appears

When you run the WandB login cell, the prompt does not appear inside the notebook. It appears in the **search bar at the top of the Codespaces window** : the bar that normally shows the file name.

When it appears:

1. Select option **(2) Use an existing W&B account**
2. Paste your API key when asked

See `wandb_login_location.png` in the repo for a screenshot showing exactly where to look.

---

*If anything goes wrong, tell your coach. There is a fallback.*

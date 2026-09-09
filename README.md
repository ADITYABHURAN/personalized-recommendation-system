# Personalized Content & Product Recommendation System (Proof of Concept)

A self-directed prototype exploring how a subscription business could move from sending
the same message to every customer toward individualized, behavior-driven recommendations.

## Why I built this

I got curious about the personalization work happening in the DTC subscription space,
particularly the shift many companies are making away from one-size-fits-all marketing
toward messaging tailored to individual customer behavior (purchase patterns, tenure,
channel, preferences). This notebook is my own exploration of what a technical approach
to that problem could look like end-to-end.

**Note:** All data used here is synthetic, generated to resemble a plausible DTC
subscription + retail business. It does not reflect any real company's actual data,
models, or results. This is a proof of concept built to demonstrate approach and
technical fluency, not a production system.

## What's inside

The notebook (`butcherbox_personalization_project.ipynb`) walks through five steps:

1. **Synthetic data generation** — ~8,000 simulated customers with subscription behavior,
   box customization patterns, channel mix, and delivery history.
2. **Feature engineering** — RFM-style features (recency, frequency, monetary) plus a
   composite churn-risk signal.
3. **Customer segmentation** — KMeans clustering to group customers by behavior before
   personalizing anything.
4. **Recommendation-response model** — an XGBoost classifier predicting how likely a
   customer is to respond to a given message positioning (utility-led, value-led, or
   premium-led), evaluated with ROC-AUC.
5. **Contextual bandit simulation** — a simple epsilon-greedy bandit that learns, per
   segment, which message positioning performs best over time, and shifts traffic toward
   it while still exploring.

## How to run it

```bash
pip install jupyter pandas numpy scikit-learn xgboost matplotlib
jupyter notebook butcherbox_personalization_project.ipynb
```

Outputs (metrics, charts) are already saved in the notebook, so it can also be read
top-to-bottom without re-running.

## What I'd explore next

- Replace the epsilon-greedy bandit with Thompson sampling or a decaying exploration
  rate, so the system explores less as confidence in a segment grows.
- Handle cold-start segments (new customers with no response history yet).
- Blend subscription and retail-channel signals more deeply as a feature source.

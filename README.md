# Personalized Content & Product Recommendation System 

A self-directed prototype exploring how a subscription business could move from sending the same message to every customer toward more individualized, behavior-driven recommendations.

## Why I built this

I built this to work through what a technical approach to that problem could look like end to end.

## What's inside

The notebook (`personalization_project.ipynb`) covers five steps:

1. Synthetic data generation: about 8,000 simulated customers with subscription behavior, box customization patterns, channel mix, and delivery history.
2. 2. Feature engineering: RFM-style features (recency, frequency, monetary) plus a composite churn-risk signal.
   3. 3. Customer segmentation: KMeans clustering to group customers by behavior before personalizing anything.
      4. 4. Recommendation-response model: an XGBoost classifier predicting how likely a customer is to respond to a given message positioning (utility-led, value-led, or premium-led), evaluated with ROC-AUC.
         5. 5. Contextual bandit simulation: a simple epsilon-greedy bandit that learns, per segment, which positioning performs best over time and shifts traffic toward it while still exploring.
           
            6. ## How to run it
           
            7. ```
               pip install jupyter pandas numpy scikit-learn xgboost matplotlib
               jupyter notebook personalization_project.ipynb
               ```

               Outputs (metrics, charts) are already saved in the notebook, so it can also just be read top to bottom without re-running anything.

               ## What I'd explore next

               - Swap the epsilon-greedy bandit for Thompson sampling or a decaying exploration rate, so it explores less as confidence in a segment grows.
               - - Handle cold-start segments (new customers with no response history yet).
                 - - Blend subscription and retail-channel signals more deeply as a feature source.
                   - 

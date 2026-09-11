# Customer Churn Prediction

Predicting which telecom customers are about to walk out the door, before they actually do.

## What this is

A notebook-based project built on the classic Telco Customer Churn dataset. It walks through cleaning messy billing data, digging into *why* people leave, and testing a handful of classic ML models to see which one calls it best.

No deep learning, no fancy pipelines — just pandas, plotly, and scikit-learn doing what they do well.

## The data

`Tel_Customer_Churn_Dataset.csv` — one row per customer, with account details (tenure, contract type, payment method), services subscribed (internet, streaming, tech support), and billing info (monthly and total charges), plus the label: did they churn or not.

A few things needed fixing before any of it was usable:

- `TotalCharges` had blank strings hiding where numbers should be — dropped those rows and cast the column to float.
- Categories like `"No internet service"` were really just another flavor of `"No"` — collapsed them to keep things simple.
- `Churn` itself was stored as Yes/No text — remapped to 1/0.

An automated profiling report (`pandas_profiling`) was run early on to get a fast first read on distributions, missing values, and correlations before writing a single plot by hand.

## Digging into the "why"

Before touching a model, the notebook spends time asking what actually predicts churn:

- Churn rate by gender, tech support, internet service, payment method, and contract length — mostly bar charts comparing group means.
- Churn rate against tenure, plotted as a scatter — the relationship is visibly not a straight line.
- An overall churn split, just to see how imbalanced the problem is.

Contract length and tenure turn out to matter a lot more than gender does — no surprise, but nice to confirm before assuming anything.

## Getting it model-ready

- One-hot encoded the categorical columns (contract, payment method, internet service, and the rest), dropping the first level of each to avoid redundancy.
- Standard-scaled `tenure`, `MonthlyCharges`, and `TotalCharges` so they sit on comparable ranges.
- Split 70/30 into train and test sets.

## Models tried

Five classifiers, trained back to back on the same split, compared purely on accuracy:

- Logistic Regression
- Support Vector Machine (linear kernel)
- K-Nearest Neighbors
- Decision Tree
- Random Forest

Logistic Regression came out on top, which is a little anticlimactic after training a random forest, but it happens — the relationships in this dataset are mostly linear enough that a simple model captures them fine.

A confusion matrix was pulled for the logistic regression model specifically, since it's the one that ended up being used going forward.

## The actual point

Accuracy is nice, but the useful output here is a probability — for every customer in the dataset, `predict_proba` gives a churn likelihood, not just a yes/no. That's what a retention team would actually act on: a ranked list of customers worth calling before they leave, not a coin flip.

## Requirements

```
pandas
numpy
matplotlib
seaborn
plotly
pandas-profiling
scikit-learn
```

## Running it

Drop `Tel_Customer_Churn_Dataset.csv` in the same folder as the notebook and run the cells top to bottom. The profiling report and plotly charts render inline.

## Where this could go next

- Cross-validation instead of a single train/test split — one split, one random_state, isn't much to trust.
- Precision and recall matter more than accuracy here, since churners are the minority class.
- A probability threshold tuned for the business cost of a missed churner vs. a wasted retention offer, rather than the default 0.5.

"""
diagnose.py

Answers one question: why does predict_fault return low confidence?

Run:  python diagnose.py

It checks, in order:
  1. What the saved model was actually trained on.
  2. Whether the model is confident on REAL dataset rows
     (isolates "model is broken" from "input is weird").
  3. What the model does with the exact hand-typed optimize.py input.
  4. How far that hand-typed input sits from real data (out-of-distribution test).
"""

import numpy as np
import pandas as pd
import joblib

from features import FEATURES

CSV = "fault_dataset_60000.csv"
PKL = "rf_fault_classifier.pkl"

df = pd.read_csv(CSV)
model = joblib.load(PKL)
model_feats = list(model.feature_names_in_)

print("=" * 64)
print("1) MODEL")
print("=" * 64)
print("Trained on features :", model_feats)
print("n_features_in_      :", getattr(model, "n_features_in_", "?"))
print("Fault classes       :", list(model.classes_), f"({len(model.classes_)} types)")
print("Random-guess floor  :", round(1 / len(model.classes_), 4))
print("features.py declares:", FEATURES)
print("Dataset columns     :", list(df.columns))
print("fault_current_ka in dataset:", "fault_current_ka" in df.columns)

print("\n" + "=" * 64)
print("2) IS THE MODEL HEALTHY ON REAL, IN-DISTRIBUTION ROWS?")
print("=" * 64)
sample = df.sample(min(2000, len(df)), random_state=0)
topconf = model.predict_proba(sample[model_feats]).max(axis=1)
print(f"Confidence on real rows -> mean {topconf.mean():.4f} | "
      f"min {topconf.min():.4f} | max {topconf.max():.4f}")
print(">> mean ~0.99  => model is fine; the problem is the INPUT you feed it.")
print(">> mean also low => the retrain itself is wrong (feature set too weak).")

print("\n" + "=" * 64)
print("3) THE EXACT HAND-TYPED optimize.py INPUT")
print("=" * 64)
query = {
    "fault_bus": 5,
    "fault_impedance_ohm": 0.25,
    "pre_fault_voltage_pu": 1.02,
    "fault_voltage_pu": 0.64,
    "loading_pu": 0.85,
    "fault_current_ka": 5.8,
}
X = pd.DataFrame([query]).reindex(columns=model_feats)
p = model.predict_proba(X)[0]
order = np.argsort(p)[::-1]
print(f"Predicted: {model.classes_[order[0]]}  (confidence {p[order[0]]:.4f})")
print("Full probability spread:")
for i in order:
    print(f"    {str(model.classes_[i]):>14} : {p[i]:.4f}")
print("A flat spread across several classes = the forest is guessing.")

print("\n" + "=" * 64)
print("4) IS THAT INPUT ACTUALLY IN-DISTRIBUTION?")
print("=" * 64)
mu = df[model_feats].mean()
sd = df[model_feats].std().replace(0, 1)
zq = ((pd.Series(query)[model_feats] - mu) / sd).values
zd = (df[model_feats] - mu) / sd
dist = np.sqrt(((zd.values - zq) ** 2).sum(axis=1))
k = 5
nearest = np.argsort(dist)[:k]
print(f"Standardized distance to {k} nearest real rows:",
      [round(float(dist[i]), 2) for i in nearest])
print("(large distances vs. typical row spacing => out of distribution)")
print("\nNearest real rows + their true fault_type:")
print(df.iloc[nearest][model_feats + ["fault_type"]].to_string(index=False))
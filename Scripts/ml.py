print("Starting the machine learning model training and evaluation...")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
df = pd.read_csv("fault_dataset_60000.csv")
df = df.drop(['row_id'], axis=1)
X = df.drop(['fault_type'], axis=1)
y= df['fault_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
probabilities = model.predict_proba(X_test)
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Confidence": probabilities.max(axis=1)
})

print(results.head())

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("Feature Importance:\n", importance)

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
importance.plot(kind='bar', x='Feature', y='Importance', legend=False)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")   
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("shap_summary_plot.png")
plt.show()


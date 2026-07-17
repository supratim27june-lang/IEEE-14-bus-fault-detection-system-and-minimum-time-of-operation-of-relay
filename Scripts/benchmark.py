import pandas as pd

import time

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("fault_dataset_60000.csv")

df = df.drop(columns=["row_id"])

X = df.drop(columns=["fault_type"])

y = df["fault_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier()
}

print("="*70)

print("MODEL COMPARISON")

print("="*70)

results = []

for name, model in models.items():

    start = time.time()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    end = time.time()

    acc = accuracy_score(y_test, pred)

    results.append([
        name,
        acc,
        end-start
    ])

results = sorted(results, key=lambda x:x[1], reverse=True)

print()

print("{:<25}{:<15}{:<15}".format(
    "Model",
    "Accuracy",
    "Time(s)"
))

print("-"*60)

for r in results:

    print("{:<25}{:<15.5f}{:<15.3f}".format(
        r[0],
        r[1],
        r[2]
    ))
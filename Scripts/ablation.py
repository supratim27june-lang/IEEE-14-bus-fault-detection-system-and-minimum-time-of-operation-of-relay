import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

df = pd.read_csv("fault_dataset_hard.csv")

df = df.drop(columns=["row_id"])

target = "fault_type"

features = [c for c in df.columns if c != target]

print("="*60)

print("Baseline Model")

print("="*60)

X = df[features]

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

baseline = accuracy_score(y_test, pred)

print("Baseline Accuracy:", baseline)

print("\n")

print("="*60)

print("Ablation Results")

print("="*60)

results = []

for feature in features:

    X = df.drop(columns=[target, feature])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results.append((feature, acc))

results = sorted(results, key=lambda x: x[1])

print("\nFeature Removed\tAccuracy")

for r in results:
    print(r[0], "\t", round(r[1],5))
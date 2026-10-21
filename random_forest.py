"""Predict student completion with a scikit-learn random forest."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------- Settings to change ----------------------------
DATA_FILE = Path(__file__).parent / "StudentChurn.csv"
TARGET = "Churn"
FEATURES = ["Age", "Line", "Grade", "Distance", "StudyGroup"]
LINE_CODES = {"EUX": 0, "HF": 1, "HHX": 2, "HTX": 3, "STX": 4}
TEST_SIZE = 0.2
RANDOM_STATE = 0
N_ESTIMATORS = 100  # Number of decision trees in the forest.
# -----------------------------------------------------------------------------


def main():
    data = pd.read_csv(DATA_FILE, sep=";")
    missing_columns = [column for column in [TARGET, *FEATURES] if column not in data]
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {', '.join(missing_columns)}")

    # Rows without a known answer cannot be used for supervised learning.
    data = data.dropna(subset=[TARGET])
    if not data[TARGET].isin(["Completed", "Stopped"]).all():
        raise ValueError("Churn must contain only Completed or Stopped.")
    if data[TARGET].nunique() != 2:
        raise ValueError("Training requires both Completed and Stopped students.")
    if not data["Line"].isin(LINE_CODES).all():
        raise ValueError("Line must contain only EUX, HF, HHX, HTX or STX.")
    data["Line"] = data["Line"].map(LINE_CODES)
    X = data[FEATURES]
    y = data[TARGET].map({"Stopped": 0, "Completed": 1})

    model = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("random_forest", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # predict_proba returns one column per class: 0 = stopped, 1 = completed.
    completion_probabilities = model.predict_proba(X_test)[:, 1]
    assert len(completion_probabilities) == len(y_test)
    assert ((completion_probabilities >= 0) & (completion_probabilities <= 1)).all()

    print(f"Features used: {', '.join(FEATURES)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(f"Majority-class baseline: {y_test.value_counts(normalize=True).max():.2%}")
    print("Confusion matrix (rows = actual, columns = predicted; order: 0, 1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Precision and recall:")
    print(classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Stopped", "Completed"], zero_division=0,
    ))

if __name__ == "__main__":
    main()

# Student completion neural network

`logistic_regression.py` uses LogisticRegression with Age, Line, Grade, Distance and StudyGroup. It keeps the same numeric mappings, stratified test split and metrics as the random forest script. Its pipeline fills missing values and scales inputs using only training data. Run `.venv/bin/python logistic_regression.py`; test with `.venv/bin/python -m unittest test_logistic_regression -v`.

The separate `random_forest.py` script uses a 100-tree RandomForestClassifier with Age, Line, Grade, Distance and StudyGroup. It keeps the same numeric mappings, stratified test split and printed metrics as `main.py`. Missing values are filled using the training median; scaling is unnecessary for trees. Run it with `.venv/bin/python random_forest.py` and test it with `.venv/bin/python -m unittest test_random_forest -v`.

Run from this directory:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
.venv/bin/python -m unittest -v
```

`main.py` follows the Titanic project's structure:

1. Read the CSV (this file uses semicolons).
2. Use Line, Grade and Distance as inputs. Ignore Id because it identifies a student rather than describing their studies.
3. Convert Churn into the answer: Stopped = 0, Completed = 1. Convert Line into a number: EUX = 0, HF = 1, HHX = 2, HTX = 3, STX = 4. Missing or unknown lines are rejected.
4. Reserve 20% of students for testing, keeping both outcomes in similar proportions.
5. Fill missing numeric values with the training median and scale all three inputs, including the numeric Line codes. The pipeline learns preprocessing only from training students.
6. Train MLPClassifier using the settings at the top of `main.py`. See [experiment_log.md](experiment_log.md) for the tuning method and results.
7. Print test accuracy, the majority-class baseline, a confusion matrix and precision/recall. Matrix order is Stopped, Completed.
8. Compute completion probability estimates for the test students and check their range. `predict_proba(... )[:, 1]` selects the probability of class 1 (Completed). Churn probability is `1 - completion_probability`.

Each run trains the model again; no model file is saved.

The Line codes are arbitrary categories, but the network treats them as ordered numbers. This is simpler to follow but may affect prediction quality compared with separate columns for each line.

Only use inputs available when making a prediction. In particular, confirm that Grade is measured before the outcome. The probability estimates have not been calibrated or validated for decisions about students. Tuning selects the best settings tested, not a guaranteed global optimum.

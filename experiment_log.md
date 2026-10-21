# Neural network tuning

Date: 2026-10-21. Objective: overall classification accuracy, the stated default while awaiting a metric preference.

Only existing neural network settings were searched. Features stayed Line, Grade and Distance; mappings, preprocessing, TEST_SIZE=0.2 and RANDOM_STATE=0 stayed fixed. The network's remaining defaults were unchanged. No new dependencies or runtime tuning were added.

## Method

- Split 1,086 rows into the original 868 training students and 218 test students.
- Compare the original settings with 72 configurations sampled using sklearn.model_selection.ParameterSampler(random_state=42).
- Initial parameter space: hidden_layer_sizes = [(4,), (8,), (16,), (32,), (64,), (8, 4), (16, 8), (32, 16), (16, 16), (8, 10, 10)]; alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]; max_iter = [500, 1000, 2000]. ParameterSampler receives two dictionaries in this order: the preceding space with solver=['adam', 'sgd'], early_stopping=[False, True]; then that space with solver=['lbfgs'], early_stopping=[False]. LBFGS does not use early stopping.
- Evaluate using StratifiedKFold(n_splits=5, shuffle=True, random_state=42), fitting imputation and scaling separately within each training fold.
- Refine using layers [(16, 8), (32, 16), (32, 8), (32, 32), (64, 32), (8, 10, 10)] crossed with alpha [0.0001, 0.001, 0.01, 0.1, 1.0], with adam, 1000 iterations and no early stopping. Skip configurations already evaluated. This adds 28 configurations, for 101 unique configurations in total.
- Re-evaluate the top eight configurations and the original baseline on five additional folds with random_state=17. Select the highest mean accuracy across both sets of five folds. This is validation for model selection, not an independent performance estimate.
- Evaluate the selected settings once on the original test partition, which was excluded from this search. This test partition has been used in earlier project experiments, so it is not a fresh external dataset. Do not choose further settings based on this result.

## Selected settings

```python
HIDDEN_LAYERS = (64, 32)
MAX_ITERATIONS = 1000
EARLY_STOPPING = False
ALPHA = 1.0
SOLVER = "adam"
```

More neurons allow a more flexible relationship between inputs and outcomes. Increasing alpha strengthens the penalty on large weights to limit overfitting. The original settings were layers=(8, 10, 10), alpha=0.0001, with the same iteration limit, early stopping and solver.

| Settings | Repeated-validation accuracy | Fixed-test accuracy |
| --- | ---: | ---: |
| Original | 76.50% | 78.44% (171/218) |
| Selected | 77.77% | 77.06% (168/218) |

The selected configuration improved validation accuracy by 1.27 percentage points but classified three fewer test students correctly. This is not evidence of a reliable improvement in future performance. Its validation fold accuracy standard deviation was 2.55 percentage points; folds overlap and this is not a confidence interval.

The selected test confusion matrix is [[19, 45], [5, 149]], ordered Stopped, Completed. It detects only 19 of 64 students who stopped (29.69% recall), despite 96.75% recall for completion. Accuracy optimization favors the majority class. Probability calibration was not performed, and no global optimum is claimed. Some search fits reached iteration limits; warnings were suppressed only in the temporary search process. The final model run produced no convergence warning.

## Finalist validation results

All finalists use early_stopping=False. Scores below average ten validation folds, not the test set.

| Hidden layers | Alpha | Solver | Max iterations | Accuracy | Dropout recall | Log loss |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| (64, 32) | 1.0 | adam | 1000 | 77.77% | 36.78% | 0.5186 |
| (64, 32) | 0.1 | adam | 1000 | 77.36% | 40.14% | 0.5296 |
| (32, 16) | 0.1 | adam | 1000 | 77.30% | 39.36% | 0.5234 |
| (32, 32) | 1.0 | adam | 1000 | 77.30% | 36.39% | 0.5182 |
| (32,) | 0.01 | adam | 1000 | 77.25% | 36.97% | 0.5197 |
| (32, 16) | 0.0001 | adam | 1000 | 77.19% | 39.36% | 0.5310 |
| (32, 16) | 0.001 | adam | 1000 | 77.02% | 38.96% | 0.5305 |
| (16,) | 0.001 | lbfgs | 500 | 76.79% | 38.76% | 0.5553 |
| (8, 10, 10) | 0.0001 | adam | 1000 | 76.50% | 37.18% | 0.5335 |

All candidate settings and per-fold scores are recorded in [tuning_results.csv](tuning_results.csv). The finalist rows contain the original five scores followed by the five additional scores; they are not ten entirely new fits.

Environment: pandas 3.0.6, scikit-learn 1.9.1, NumPy 2.5.4. CSV SHA-256: `a0036e8e6f1009ece69c5d9da56cf85491de6a46f60735f9eecb0242a068b790`.

## Checks

The final script ran successfully and all nine unittest tests passed. The existing main test was updated to check the current classification report instead of the previously removed per-student probability printout. Assertions checked that training and test indices were disjoint during tuning and that predicted probabilities are within [0, 1] in the scripts.

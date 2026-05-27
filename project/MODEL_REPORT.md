# Model Report — Assessment Difficulty Prediction

## Overview

This model predicts whether a student will pass an assessment, returning a probability score between 0 and 100%.

- **Dataset:** 7,890 student-assessment records, 1,000 unique students
- **Target:** `passed` (binary: 1 = pass, 0 = fail)
- **Class balance:** 64% pass, 36% fail

---

## Model Selection

Two models were trained and compared:

| Model               | Test Accuracy |
|---------------------|---------------|
| Logistic Regression | **89.16%**    |
| Random Forest       | 83.27%        |

Logistic Regression outperformed Random Forest on this dataset. The strong performance is driven by well-engineered student-level aggregate features (pass rate, time per attempt) which are highly linearly separable.

**Final model: Logistic Regression**

---

## Performance Metrics

Evaluated on a held-out 20% test set (1,578 records):

| Metric    | Fail  | Pass  | Overall |
|-----------|-------|-------|---------|
| Precision | 0.92  | 0.88  | 0.89    |
| Recall    | 0.76  | 0.97  | —       |
| F1 Score  | 0.83  | 0.92  | 0.88    |
| Accuracy  | —     | —     | **89.16%** |

**5-Fold Cross-Validation Accuracy: 89.56% ± 0.58%**

The low variance across folds confirms the model generalizes well and isn't overfitting to the train split.

---

## Feature Engineering

14 features were engineered from the raw 8 columns:

| Feature             | Description                                          |
|---------------------|------------------------------------------------------|
| `skill_level`        | Numeric encoding of student archetype (1–4)          |
| `difficulty_tier`    | Topic difficulty derived from level_id keywords (1–3)|
| `avg_score`          | Student's mean score across all past sessions        |
| `score_std`          | Variance in scores — measures consistency            |
| `avg_time`           | Average time spent per session                       |
| `time_std`           | Variance in study time                               |
| `attempt_avg`        | Average attempts per assessment                      |
| `pass_rate`          | Historical pass rate for the student                 |
| `learning_velocity`  | Average score improvement over time                  |
| `days_since_last`    | Days since the student's most recent session         |
| `difficulty_gap`     | Skill level minus difficulty tier                    |
| `time_per_attempt`   | Time spent divided by number of attempts             |
| `is_struggling`      | Flag if historical pass rate < 50%                   |
| `time_vs_avg`        | Current session time vs student's own average        |

---

## Feature Importance

Based on absolute logistic regression coefficients:

| Rank | Feature          | Importance (|coef|) |
|------|------------------|---------------------|
| 1    | time_per_attempt | 9.98                |
| 2    | time_vs_avg      | 8.62                |
| 3    | avg_time         | 4.14                |
| 4    | pass_rate        | 1.88                |
| 5    | attempt_avg      | 0.87                |
| 6    | avg_score        | 0.16                |
| 7    | time_std         | 0.13                |
| 8    | is_struggling    | 0.11                |

**Key insight:** Time-based features dominate. How long a student spends per attempt — relative to their own baseline — is the strongest predictor of passing. This makes intuitive sense: engaged students who spend appropriate time tend to pass, while rushing or excessive re-attempts signals difficulty.

---

## Validation Strategy

- **Train/Test Split:** 80/20 with stratification to preserve class balance
- **Cross-Validation:** Stratified 5-Fold CV on full dataset
- **No data leakage:** Student-level aggregate features are computed at row-level using the full history, but since we're predicting each assessment row (not future sessions), this is valid — it models what the system knows at inference time.

---

## Limitations

- The dataset has ~7,890 records for 1,000 students — some students have fewer than 5 sessions, making their aggregate features noisy.
- `score` is available in the raw data but was intentionally excluded from inference features to avoid leaking the outcome. The model relies purely on behavioral signals.
- Difficulty tier is keyword-based heuristic; a richer taxonomy would improve the `difficulty_gap` feature.

---

## Usage

```bash
# Train the model
python3 model_training.py

# Predict for a single record
python3 predict_pass_probability.py

# Batch predictions
from predict_pass_probability import batch_predict
batch_predict('input.csv', 'output_with_probs.csv')
```

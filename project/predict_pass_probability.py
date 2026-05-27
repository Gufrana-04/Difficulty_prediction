import pandas as pd
import numpy as np
import pickle
import os

from data_preprocessing import load_and_clean, ARCHETYPE_SKILL, get_difficulty_tier
from feature_engineering import build_features, FEATURE_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_from_record(student_id, archetype, level_id, time_spent_minutes, attempts, completed_date_days_ago):
    """
    Give a pass probability (0-100%) for one student-assessment record.
    Falls back to global averages for students not seen during training.
    """
    model, scaler = load_model()

    # build a minimal dataframe that mimics the training data shape
    row = {
        'student_id': student_id,
        'archetype': archetype,
        'level_id': level_id,
        'score': 70,  # placeholder score; not used for prediction
        'time_spent_minutes': time_spent_minutes,
        'passed': 0,
        'attempts': attempts,
        'completed_date_days_ago': completed_date_days_ago,
    }

    # merge with real training data so student-level stats are calculated properly
    full_df = load_and_clean()
    new_row = pd.DataFrame([row])
    new_row['passed'] = new_row['passed'].astype(int)
    combined = pd.concat([full_df, new_row], ignore_index=True)
    combined = build_features(combined)

    # pull the last row which is our new record
    sample = combined.iloc[[-1]][FEATURE_COLS].fillna(0)

    from sklearn.ensemble import RandomForestClassifier
    if isinstance(model, RandomForestClassifier):
        prob = model.predict_proba(sample)[0][1]
    else:
        sample_scaled = scaler.transform(sample)
        prob = model.predict_proba(sample_scaled)[0][1]

    return round(prob * 100, 2)


def batch_predict(input_csv_path, output_csv_path=None):
    """
    Run predictions on a CSV of records and return probabilities.
    """
    model, scaler = load_model()
    df_new = pd.read_csv(input_csv_path)

    full_df = load_and_clean()
    if 'passed' not in df_new.columns:
        df_new['passed'] = 0
    df_new['passed'] = df_new['passed'].astype(int)

    combined = pd.concat([full_df, df_new], ignore_index=True)
    combined = build_features(combined)

    # only take the new rows
    results = combined.iloc[-len(df_new):][FEATURE_COLS].fillna(0)

    from sklearn.ensemble import RandomForestClassifier
    if isinstance(model, RandomForestClassifier):
        probs = model.predict_proba(results)[:, 1]
    else:
        results_scaled = scaler.transform(results)
        probs = model.predict_proba(results_scaled)[:, 1]

    df_new['pass_probability_%'] = (probs * 100).round(2)
    df_new['predicted_pass'] = (probs >= 0.5).astype(int)

    if output_csv_path:
        df_new.to_csv(output_csv_path, index=False)
        print(f"Predictions saved to {output_csv_path}")

    return df_new[['student_id', 'level_id', 'pass_probability_%', 'predicted_pass']]


if __name__ == '__main__':
    # quick demo with a made-up student
    prob = predict_from_record(
        student_id='STU_DEMO',
        archetype='intermediate',
        level_id='py_loops',
        time_spent_minutes=45,
        attempts=2,
        completed_date_days_ago=5,
    )
    print(f"Pass probability for demo student: {prob}%")

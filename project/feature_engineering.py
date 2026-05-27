import pandas as pd
import numpy as np
from data_preprocessing import load_and_clean


def build_features(df):
    # per-student aggregate stats from their full history
    student_stats = df.groupby('student_id').agg(
        avg_score=('score', 'mean'),
        score_std=('score', 'std'),          # how consistent the student is
        total_time=('time_spent_minutes', 'sum'),
        avg_time=('time_spent_minutes', 'mean'),
        time_std=('time_spent_minutes', 'std'),  # consistency in study time
        attempt_avg=('attempts', 'mean'),
        pass_rate=('passed', 'mean'),
        total_attempts=('attempts', 'sum'),
        sessions=('level_id', 'count'),
    ).reset_index()

    student_stats['score_std'] = student_stats['score_std'].fillna(0)
    student_stats['time_std'] = student_stats['time_std'].fillna(0)

    df = df.merge(student_stats, on='student_id', how='left')

    # learning velocity: how much the score improves over time per student
    df = df.sort_values(['student_id', 'completed_date_days_ago'], ascending=[True, False])
    df['score_lag'] = df.groupby('student_id')['score'].shift(1)
    df['score_delta'] = df['score'] - df['score_lag']
    df['learning_velocity'] = df.groupby('student_id')['score_delta'].transform('mean')
    df['learning_velocity'] = df['learning_velocity'].fillna(0)

    # how long since the last session (recency)
    df['days_since_last'] = df.groupby('student_id')['completed_date_days_ago'].transform('min')

    # difficulty gap: difference between student skill and question difficulty
    df['difficulty_gap'] = df['skill_level'] - df['difficulty_tier']

    # average time per attempt - efficiency measure
    df['time_per_attempt'] = df['time_spent_minutes'] / df['attempts'].replace(0, 1)

    # whether this student tends to struggle (low historical pass rate)
    df['is_struggling'] = (df['pass_rate'] < 0.5).astype(int)

    # how much time the student spent vs their own average (effort signal)
    df['time_vs_avg'] = df['time_spent_minutes'] - df['avg_time']

    return df


FEATURE_COLS = [
    'skill_level',
    'difficulty_tier',
    'avg_score',
    'score_std',
    'avg_time',
    'time_std',
    'attempt_avg',
    'pass_rate',
    'learning_velocity',
    'days_since_last',
    'difficulty_gap',
    'time_per_attempt',
    'is_struggling',
    'time_vs_avg',
]

TARGET_COL = 'passed'


if __name__ == '__main__':
    df = load_and_clean()
    df = build_features(df)
    print(f"Feature matrix shape: {df[FEATURE_COLS].shape}")
    print("Sample features:")
    print(df[FEATURE_COLS].head())
    print("\nNo nulls in features:", df[FEATURE_COLS].isnull().sum().sum() == 0)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'student_progress.csv')

# maps archetype strings to a numeric skill level
ARCHETYPE_SKILL = {
    'beginner': 1,
    'intermediate': 2,
    'math_focused': 3,
    'code_focused': 3,
    'advanced': 4,
}

# rough difficulty of a topic based on keywords in the level_id
def get_difficulty_tier(level_id):
    level_id = str(level_id).lower()
    if any(k in level_id for k in ['algo', 'oop', 'dp', 'graph', 'tree']):
        return 3  # hard
    elif any(k in level_id for k in ['functions', 'loops', 'dicts', 'linear', 'probability']):
        return 2  # medium
    else:
        return 1  # easy


def load_and_clean():
    df = pd.read_csv(DATA_PATH)

    # drop duplicates just in case
    df = df.drop_duplicates()

    # encode archetype as skill level number
    df['skill_level'] = df['archetype'].map(ARCHETYPE_SKILL).fillna(2).astype(int)

    # encode difficulty of the question/level
    df['difficulty_tier'] = df['level_id'].apply(get_difficulty_tier)

    # normalize passed column to int
    df['passed'] = df['passed'].astype(int)

    # encode level_id as integer for use if needed
    le = LabelEncoder()
    df['level_encoded'] = le.fit_transform(df['level_id'])

    df = df.reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = load_and_clean()
    print(f"Loaded {len(df)} records, {df['student_id'].nunique()} unique students")
    print(df[['student_id', 'skill_level', 'difficulty_tier', 'passed']].head())

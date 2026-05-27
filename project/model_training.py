import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from data_preprocessing import load_and_clean
from feature_engineering import build_features, FEATURE_COLS, TARGET_COL

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')


def train():
    df = load_and_clean()
    df = build_features(df)

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale for logistic regression but keep raw for RF
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # logistic regression as baseline
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)

    # random forest - usually stronger for this kind of tabular data
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Random Forest Accuracy:       {rf_acc:.4f}")

    # pick the better model
    if rf_acc >= lr_acc:
        best_model = rf
        best_preds = rf_preds
        best_X_test = X_test
        print("\nUsing Random Forest")
    else:
        best_model = lr
        best_preds = lr_preds
        best_X_test = X_test_scaled
        print("\nUsing Logistic Regression")

    print("\nClassification Report:")
    print(classification_report(y_test, best_preds, target_names=['Fail', 'Pass']))

    # cross-validation on full dataset to confirm no overfitting
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if best_model is rf:
        cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    else:
        from sklearn.pipeline import make_pipeline
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')

    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # feature importance (RF gives this directly)
    if best_model is rf:
        importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
        importances = importances.sort_values(ascending=False)
        print("\nTop Feature Importances:")
        print(importances.to_string())

    # save model and scaler for inference
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nModel saved to {MODEL_PATH}")

    return best_model, scaler, {
        'accuracy': accuracy_score(y_test, best_preds),
        'precision': precision_score(y_test, best_preds),
        'recall': recall_score(y_test, best_preds),
        'f1': f1_score(y_test, best_preds),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importances': importances.to_dict() if best_model is rf else {},
    }


if __name__ == '__main__':
    train()

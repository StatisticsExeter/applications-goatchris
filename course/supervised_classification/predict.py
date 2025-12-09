import joblib
import pandas as pd
from course.utils import find_project_root


def predict(model_path, X_test_path, y_pred_path, y_pred_prob_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    """Form an object y_pred containing a list of your classifer predictions"""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame({"predicted_built_age": y_pred})
    y_pred_df.to_csv(y_pred_path, index=False)

    y_scores = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_scores = proba[:, 1]
        else:
            y_scores = proba.ravel()
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_scores = scores if scores.ndim == 1 else scores[:, 0]
    else:
        y_scores = [0.5] * len(y_pred)

    y_pred_prob_df = pd.DataFrame({"predicted_built_age": y_scores})
    y_pred_prob_df.to_csv(y_pred_prob_path, index=False)


def pred_lda():
    base_dir = find_project_root()
    model_path = base_dir / "data_cache" / "models" / "lda_model.joblib"
    X_test_path = base_dir / "data_cache" / "energy_X_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "lda_y_pred.csv"
    y_pred_prob_path = base_dir / "data_cache" / "models" / "lda_y_pred_prob.csv"
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)


def pred_qda():
    base_dir = find_project_root()
    model_path = base_dir / "data_cache" / "models" / "qda_model.joblib"
    X_test_path = base_dir / "data_cache" / "energy_X_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "qda_y_pred.csv"
    y_pred_prob_path = base_dir / "data_cache" / "models" / "qda_y_pred_prob.csv"
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from typing import Literal, Tuple, Dict, Any


def run_classification(
    df: pd.DataFrame,
    target_column: str,
    model_type: Literal["tree", "logistic", "random_forest", "svm"] = "tree",
    test_size: float = 0.2,
    random_state: int = 42) -> Tuple[str, Dict[str, Any]]:
    """
    Train and evaluate a classifier.
 
    Parameters
    ----------
    df            : fully preprocessed DataFrame
    target_column : name of the label column
    model_type    : 'tree' (Decision Tree) or 'logistic' (Logistic Regression) or 'random_forest' (Random Forest) or 'svm' (Linear SVM (LinearSVC))
    test_size     : fraction of data reserved for testing
    random_state  : reproducibility seed
 
    Returns
    -------
    summary : str
        A short human-readable summary of results (for agent response).
    details : dict
        Full structured results (for report generation).
    """
    try:
        if target_column not in df.columns:
            msg = f"Error: target column '{target_column}' not found in the dataset."
            return msg, {}
 
        X = df.drop(columns=[target_column])
        y = df[target_column]
 
        if y.nunique() < 2:
            msg = "Error: the target column has fewer than 2 unique classes — cannot classify."
            return msg, {}
 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
 
        # Model selection
        if model_type == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=random_state)
            model_name = "Logistic Regression"
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_leaf=5,
                random_state=random_state, n_jobs=-1
            )
            model_name = "Random Forest"
        elif model_type == "svm":
            model = LinearSVC(max_iter=2000, random_state=random_state)
            model_name = "Linear SVM"
        else: # default: tree
            model = DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=5, random_state=random_state
            )
            model_name = "Decision Tree"
 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
 
        acc = accuracy_score(y_test, y_pred)
        clf_report_str = classification_report(y_test, y_pred, zero_division=0)
        clf_report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
 
        # Feature importance — Decision Tree and Random Forest only
        # LinearSVC and LogisticRegression don't expose feature_importances_
        top_features = {}
        if model_type in ("tree", "random_forest"):
            importances = pd.Series(model.feature_importances_, index=X.columns)
            top_features = importances.sort_values(ascending=False).head(5).to_dict()
 
        # summary (short, for agent)
        importance_lines = ""
        if top_features:
            importance_lines = "\nTop 5 features: " + ", ".join(
                f"{f}({s:.3f})" for f, s in top_features.items()
            )
 
        summary = (
            f"{model_name} — Accuracy: {acc:.4f} ({acc*100:.2f}%)"
            f" | Train: {len(X_train)} | Test: {len(X_test)}"
            f"{importance_lines}"
        )
 
        # details (full, for report)
        cm_lines = ["Confusion Matrix:"]
        for row in cm:
            cm_lines.append("  " + "  ".join(str(v).rjust(5) for v in row))
 
        details = {
            "model_name": model_name,
            "model_type": model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": acc,
            "classification_report_str": clf_report_str,
            "classification_report_dict": clf_report_dict,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_str": "\n".join(cm_lines),
            "top_features": top_features,
        }
 
        return summary, details
 
    except Exception as e:
        msg = f"Error during classification: {str(e)}"
        return msg, {}
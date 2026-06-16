import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df: pd.DataFrame, plan: dict, target_col: str) -> Tuple[pd.DataFrame, str]:
    """
    Preprocess the input DataFrame according to the specified plan.

    Parameters
    ----------
    df :            The input DataFrame to preprocess.
    plan :          A dictionary specifying the preprocessing steps to apply.
    target_col :    The name of the target column (if applicable).

    Returns
    -------
    (df, report)
        df          The preprocessed DataFrame.
        report      A human-readable report of the preprocessing steps taken.
    """
    report_lines = []
    df = df.copy()
    steps = plan

    # 1. Remove duplicates
    if steps.get("remove_duplicates", {}).get("apply"):
        before = len(df)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        dropped = before - len(df)
        if dropped:
            report_lines.append(f"Removed {dropped} duplicate row(s).")

    # 2. Handle missing values
    if steps.get("handle_missing_values", {}).get("apply"):
        strategy = steps["handle_missing_values"].get("strategy", "median")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if strategy == "median":
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
        else:  # mode for all
            for col in numeric_cols + cat_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
        # categorical missing always filled with mode
        for col in cat_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        report_lines.append(f"Missing values handled using strategy: '{strategy}'.")

    # 3. Handle outliers (before encoding, on numeric cols only)
    outlier_cfg = steps.get("handle_outliers", {})
    if outlier_cfg.get("apply"):
        method = outlier_cfg.get("method", "iqr")
        cols = [c for c in outlier_cfg.get("columns", []) if c in df.columns and c != target_col]
        for col in cols:
            if method == "iqr":
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            else:  # zscore
                mean, std = df[col].mean(), df[col].std()
                df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)
        if cols:
            report_lines.append(f"Outliers handled ({method.upper()}) for: {', '.join(cols)}.")

    # 4. Encode categoricals
    enc_cfg = steps.get("encode_categoricals", {})
    if enc_cfg.get("apply"):
        method = enc_cfg.get("method", "label")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if method == "onehot":
            cols_to_encode = [c for c in cat_cols if c != target_col]
            df = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
            # label-encode the target separately
            if target_col and target_col in df.columns:
                df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))
            report_lines.append(f"One-hot encoded {len(cols_to_encode)} column(s). Target '{target_col}' label-encoded.")
        else:  # label
            le = LabelEncoder()
            for col in cat_cols:
                df[col] = le.fit_transform(df[col].astype(str))
            report_lines.append(f"Label-encoded {len(cat_cols)} categorical column(s).")

    # 5. Scale numericals
    scale_cfg = steps.get("scale_numericals", {})
    if scale_cfg.get("apply"):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        method = scale_cfg.get("method", "standard")
        cols = [c for c in scale_cfg.get("columns", []) if c in df.columns and c != target_col]
        if cols:
            scaler = StandardScaler() if method == "standard" else MinMaxScaler()
            df[cols] = scaler.fit_transform(df[cols])
            report_lines.append(f"Scaled {len(cols)} column(s) using {method} scaler: {', '.join(cols)}.")

    df.reset_index(drop=True, inplace=True)
    summary = (
        "Preprocessing complete.\n"
        + ("\n".join(report_lines) if report_lines else "No steps were applied.")
        + f"\nFinal shape: {df.shape[0]} rows x {df.shape[1]} columns."
    )
    # df.to_csv("./data/temp_preprocessed_data.csv", index=False)  # Save preprocessed data for potential future use
    return df, summary
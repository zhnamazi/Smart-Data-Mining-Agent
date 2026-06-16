import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import Literal


def extract_patterns(
    df: pd.DataFrame,
    method: Literal["apriori", "fpgrowth"] = "fpgrowth",
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    max_rules: int = 15,
) -> str:
    """
    Mine frequent itemsets and association rules from *df*.

    Strategy
    --------
    - Each row is treated as a transaction.
    - Numeric columns are binarised at their median (above median → True).
    - Categorical columns (already int-encoded) are binarised at their median too.
    - Columns with only one unique value are dropped (they carry no information).

    Parameters
    ----------
    df               preprocessed DataFrame (target column may still be present)
    method           'apriori' or 'fpgrowth'
    min_support      minimum support threshold
    min_confidence   minimum confidence for association rules
    max_rules        maximum number of rules to report

    Returns
    -------
    A human-readable string summarising the top rules.
    """
    try:
        # Binarise
        binary_df = pd.DataFrame()
        for col in df.columns:
            if df[col].nunique() <= 1:
                continue  # constant column, skip
            median_val = df[col].median()
            binary_df[f"{col}>median"] = (df[col] > median_val).astype(bool)

        if binary_df.empty:
            return "Could not binarise the dataset for frequent-pattern mining."

        # Mine itemsets
        if method == "apriori":
            frequent_itemsets = apriori(
                binary_df, min_support=min_support, use_colnames=True
            )
        else:
            frequent_itemsets = fpgrowth(
                binary_df, min_support=min_support, use_colnames=True
            )

        if frequent_itemsets.empty:
            return (
                f"No frequent itemsets found with min_support={min_support}. "
                "Try lowering the support threshold."
            )

        # Extract rules
        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=min_confidence
        )

        if rules.empty:
            return (
                f"Found {len(frequent_itemsets)} frequent itemsets, "
                f"but no association rules met min_confidence={min_confidence}."
            )

        # Sort by lift then confidence, return top N
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
        top_rules = rules.head(max_rules)

        lines = [
            f"Frequent Pattern Mining ({method.upper()}) Results",
            f"Min support: {min_support} | Min confidence: {min_confidence}",
            f"Frequent itemsets found: {len(frequent_itemsets)}",
            f"Association rules found: {len(rules)}",
            "",
            f"Top {len(top_rules)} rules (sorted by lift):",
        ]
        for i, row in top_rules.iterrows():
            ant = ", ".join(list(row["antecedents"]))
            con = ", ".join(list(row["consequents"]))
            lines.append(
                f"  [{ant}] → [{con}]  "
                f"support={row['support']:.3f}, "
                f"confidence={row['confidence']:.3f}, "
                f"lift={row['lift']:.3f}"
            )

        summary_lines = lines[:5]
        detail_lines = lines

        summary_text = "\n".join(summary_lines)
        detail_text = "\n".join(detail_lines)
        print(detail_lines)

        return summary_text, detail_text

    except Exception as e:
        return f"Error during frequent pattern extraction: {str(e)}"
import pandas as pd
from typing import Tuple, Optional


def load_dataset(file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load a CSV file from *file_path*.

    parameters
    ----------
    file_path :     The path to the CSV file to load.

    Returns
    -------
    (df, message)
        df          the loaded DataFrame, or None on failure
        message     human-readable status string
    """
    try:
        df = pd.read_csv(file_path)
        msg = (
            f"Dataset loaded successfully from '{file_path}'.\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns.\n"
            f"Columns: {', '.join(df.columns.tolist())}"
        )
        return df, msg
    except FileNotFoundError:
        return None, f"Error: File not found at path '{file_path}'. Please check the path and try again."
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"
from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd


def results_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert benchmark result rows (list of dicts) to a pandas DataFrame.

    Expected row keys:
      - instance
      - solver
      - score
      - time_sec
      - feasible
      - gap
      - extras   (dict, will be left as-is)
    """
    return pd.DataFrame(rows)


def save_results_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Save benchmark results to a CSV file."""
    df = results_to_dataframe(rows)
    df.to_csv(path, index=False)


def load_results_csv(path: str) -> List[Dict[str, Any]]:
    """Load benchmark results from a CSV file into a list of dicts."""
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

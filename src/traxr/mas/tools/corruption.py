"""DataFrame-level corruption for tool-based experiments."""

import pandas as pd
import numpy as np
from typing import Optional
import random


class DataFrameCorruptor:
    """Applies corruption to pandas DataFrames with predictable effects.

    Corruption types:
    - Column swap: Swap two columns
    - Column rename: Rename columns (e.g., "Year" → "yr", "Title" → "Ttl")
    - Value corruption: Modify cell values (add prefixes, change case)
    - Row duplication: Duplicate rows
    - Type changes: Convert types (int → str, etc.)
    - Sort disruption: Randomly shuffle rows
    """

    def __init__(self, seed: int = 42, corruption_probability: float = 0.7):
        """Initialize DataFrame corruptor.

        Args:
            seed: Random seed
            corruption_probability: Probability of applying each corruption
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.corruption_probability = corruption_probability
        self.applied_corruptions = []

    def corrupt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply corruptions to DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Corrupted DataFrame
        """
        self.applied_corruptions = []
        corrupted_df = df.copy()

        # Apply each corruption with probability
        corrupted_df = self._maybe_apply(self._swap_columns, corrupted_df, "column_swap")
        corrupted_df = self._maybe_apply(self._rename_columns, corrupted_df, "column_rename")
        corrupted_df = self._maybe_apply(self._corrupt_values, corrupted_df, "value_corruption")
        corrupted_df = self._maybe_apply(self._duplicate_rows, corrupted_df, "row_duplication")
        corrupted_df = self._maybe_apply(self._corrupt_types, corrupted_df, "type_corruption")
        corrupted_df = self._maybe_apply(self._shuffle_rows, corrupted_df, "row_shuffle")

        return corrupted_df

    def get_corruption_description(self) -> str:
        """Get description of applied corruptions."""
        if not self.applied_corruptions:
            return "no corruption"
        return ", ".join(self.applied_corruptions)

    def _maybe_apply(self, func, df, name):
        """Apply corruption with probability."""
        if self.rng.random() < self.corruption_probability:
            self.applied_corruptions.append(name)
            return func(df)
        return df

    def _swap_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Swap two random columns."""
        if len(df.columns) < 2:
            return df

        cols = list(df.columns)
        idx1, idx2 = self.rng.sample(range(len(cols)), 2)

        cols[idx1], cols[idx2] = cols[idx2], cols[idx1]
        return df[cols]

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns with abbreviations."""
        rename_map = {
            "Title": "Ttl",
            "Year": "yr",
            "Platform": "Plat",
            "Status": "Sts",
            "Genre": "Gnr",
            "Description": "Desc",
            "Name": "Nm",
            "Date": "Dt",
            "Amount": "Amt",
            "Price": "Pr",
        }

        new_columns = {}
        for col in df.columns:
            if col in rename_map:
                new_columns[col] = rename_map[col]
            elif len(col) > 4:
                # Abbreviate to first 3-4 chars
                new_columns[col] = col[:3].lower()

        if new_columns:
            return df.rename(columns=new_columns)
        return df

    def _corrupt_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrupt cell values."""
        corrupted_df = df.copy()

        for col in corrupted_df.columns:
            if corrupted_df[col].dtype == 'object':  # String columns
                # Randomly modify some values
                mask = self.np_rng.random(len(corrupted_df)) < 0.3
                corrupted_df.loc[mask, col] = corrupted_df.loc[mask, col].apply(
                    lambda x: str(x).replace("-", "_").replace(" ", "_") if pd.notna(x) else x
                )

        return corrupted_df

    def _duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Duplicate some rows."""
        if len(df) == 0:
            return df

        n_duplicates = max(1, int(len(df) * 0.2))
        indices_to_dup = self.rng.sample(range(len(df)), min(n_duplicates, len(df)))

        duplicated_rows = []
        for idx in indices_to_dup:
            row = df.iloc[idx].copy()
            duplicated_rows.append(row)

        if duplicated_rows:
            dup_df = pd.DataFrame(duplicated_rows)
            return pd.concat([df, dup_df], ignore_index=True)

        return df

    def _corrupt_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrupt data types (e.g., numbers → strings with prefixes)."""
        corrupted_df = df.copy()

        for col in corrupted_df.columns:
            if pd.api.types.is_numeric_dtype(corrupted_df[col]):
                # Add prefix to numeric values
                mask = self.np_rng.random(len(corrupted_df)) < 0.4
                corrupted_df.loc[mask, col] = corrupted_df.loc[mask, col].apply(
                    lambda x: f"~{x}" if pd.notna(x) else x
                )

        return corrupted_df

    def _shuffle_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Randomly shuffle a portion of rows."""
        if len(df) < 2:
            return df

        # Shuffle 30-50% of rows
        n_shuffle = int(len(df) * self.rng.uniform(0.3, 0.5))
        indices_to_shuffle = self.rng.sample(range(len(df)), n_shuffle)

        shuffled_df = df.copy()
        rows_to_shuffle = shuffled_df.iloc[indices_to_shuffle]
        shuffled_indices = self.rng.sample(indices_to_shuffle, len(indices_to_shuffle))

        for orig_idx, new_idx in zip(indices_to_shuffle, shuffled_indices):
            shuffled_df.iloc[orig_idx] = rows_to_shuffle.iloc[list(indices_to_shuffle).index(new_idx)]

        return shuffled_df

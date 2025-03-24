import pandas as pd
import numpy as np
import warnings

from tempdisagg.utils.logging_utils import VerboseLogger


import pandas as pd
import numpy as np
import warnings

from tempdisagg.utils.logging_utils import VerboseLogger


class TimeSeriesCompleter:
    def __init__(self, df, index_col, grain_col, y_col, X_col,
                 interpolation_method='nearest', verbose=False):
        self.df = df.copy()
        self.index_col = index_col
        self.grain_col = grain_col
        self.y_col = y_col
        self.X_col = X_col
        self.interpolation_method = interpolation_method
        self.verbose = verbose

        self.df_full = pd.DataFrame()
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        self._validate_input()

    def complete_series(self):
        # ✅ Guardar copia original antes de alterar
        original_df = self.df.copy()

        try:
            if self.verbose:
                self.logger.info("Creating full index with all combinations...")
            self._create_full_index()
        except Exception as e:
            self.logger.error(f"Failed to create full index: {e}")
            return self.df_full, {}

        for col in [self.y_col, self.X_col]:
            try:
                if self.verbose:
                    self.logger.info(f"Imputing values in column: '{col}'...")
                self._impute_column(col)
            except Exception as e:
                self.logger.error(f"Imputation failed for '{col}': {e}")
                return self.df_full, {}

        self._validate_output_no_nans()

        # 📐 Padding detection
        original_pairs = list(zip(original_df[self.index_col], original_df[self.grain_col]))
        completed_pairs = list(zip(self.df_full[self.index_col], self.df_full[self.grain_col]))

        match_positions = [completed_pairs.index(pair) for pair in original_pairs if pair in completed_pairs]
        if not match_positions:
            raise ValueError("Original keys not found in completed DataFrame — check index/grain consistency.")

        start = min(match_positions)
        end = max(match_positions)
        n_pad_before = start
        n_pad_after = len(completed_pairs) - end - 1

        padding_info = {
            "original_length": len(original_pairs),
            "completed_length": len(completed_pairs),
            "n_pad_before": n_pad_before,
            "n_pad_after": n_pad_after
        }

        # 🧼 Reimputación de seguridad en cascada
        for col in [self.y_col, self.X_col]:
            if self.df_full[col].isna().any():
                self.df_full[col] = (
                    self.df_full[col]
                    .interpolate(method='linear', limit_direction='both')
                    .ffill().bfill()
                    .interpolate(method='nearest', limit_direction='both')
                )

        return self.df_full, padding_info

    def _validate_input(self):
        try:
            required_cols = [self.index_col, self.grain_col, self.y_col, self.X_col]
            for col in required_cols:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: '{col}'")

            for col in [self.y_col, self.X_col]:
                if self.df[col].dropna().empty:
                    raise ValueError(f"Column '{col}' contains only missing values.")

            self.df[self.index_col] = self.df[self.index_col].astype(int)
            self.df[self.grain_col] = self.df[self.grain_col].astype(int)
            self.df[self.y_col] = pd.to_numeric(self.df[self.y_col], errors='coerce')
            self.df[self.X_col] = pd.to_numeric(self.df[self.X_col], errors='coerce')

            if self.df.shape[0] < 3:
                raise ValueError("At least 3 observations are required for interpolation.")

            if self.verbose:
                self.logger.info("Input validation successful.")
        except Exception as e:
            raise ValueError(f"Input validation failed: {e}")

    def _create_full_index(self):
        try:
            all_indices = sorted(self.df[self.index_col].unique())
            all_grains = sorted(self.df[self.grain_col].unique())

            # Padding forward
            last_index = all_indices[-1]
            df_last = self.df[self.df[self.index_col] == last_index]
            missing_grains = [g for g in all_grains if g not in df_last[self.grain_col].unique()]
            if missing_grains:
                if self.verbose:
                    self.logger.info(f"Padding forward: {last_index} missing grains {missing_grains}")
                filler = pd.DataFrame({
                    self.index_col: [last_index] * len(missing_grains),
                    self.grain_col: missing_grains,
                    self.y_col: [np.nan] * len(missing_grains),
                    self.X_col: [np.nan] * len(missing_grains)
                })
                self.df = pd.concat([self.df, filler], ignore_index=True)

            # Padding backward
            first_index = all_indices[0]
            df_first = self.df[self.df[self.index_col] == first_index]
            missing_grains_first = [g for g in all_grains if g not in df_first[self.grain_col].unique()]
            if missing_grains_first:
                if self.verbose:
                    self.logger.info(f"Padding backward: {first_index} missing grains {missing_grains_first}")
                filler_start = pd.DataFrame({
                    self.index_col: [first_index] * len(missing_grains_first),
                    self.grain_col: missing_grains_first,
                    self.y_col: [np.nan] * len(missing_grains_first),
                    self.X_col: [np.nan] * len(missing_grains_first)
                })
                self.df = pd.concat([filler_start, self.df], ignore_index=True)

            all_indices_full = sorted(self.df[self.index_col].unique())
            full_index = pd.MultiIndex.from_product(
                [all_indices_full, all_grains],
                names=[self.index_col, self.grain_col]
            )

            self.df_full = (
                self.df.set_index([self.index_col, self.grain_col])
                .reindex(full_index)
                .reset_index()
            ).sort_values(by=[self.index_col, self.grain_col]).reset_index(drop=True)

            if self.verbose:
                self.logger.info(f"Completed index: {self.df_full.shape[0]} rows")

        except Exception as e:
            raise RuntimeError(f"Error while creating full index: {e}")

    def _impute_column(self, col_name):
        try:
            before = self.df_full[col_name].isna().sum()
            self.df_full[col_name] = (
                self.df_full[col_name]
                .interpolate(method=self.interpolation_method, limit_direction='both')
                .ffill().bfill()
                .interpolate(method='nearest', limit_direction='both')
            )
            after = self.df_full[col_name].isna().sum()
            if after > 0:
                warnings.warn(
                    f"Column '{col_name}' still contains {after} missing values after fallback interpolation.",
                    UserWarning
                )
            if self.verbose:
                self.logger.info(f"Column '{col_name}' imputed: {before} → {after} missing values.")
        except Exception as e:
            raise RuntimeError(f"Error during imputation of '{col_name}': {e}")

    def _validate_output_no_nans(self):
        missing = self.df_full[[self.y_col, self.X_col]].isna().sum()
        if missing.any():
            raise ValueError(f"Missing values remain after imputation:\n{missing}")
        if self.verbose:
            self.logger.info("All target columns fully imputed — no missing values remain.")

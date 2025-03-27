import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

from tempdisagg.utils.logging_utils import VerboseLogger


class Retropolarizer:
    def __init__(self, df, new_col, old_col):
        self.df = df.copy()
        self.new_col = new_col
        self.old_col = old_col

        self.df[self.new_col] = self.df[self.new_col].interpolate(method='linear')
        self.df[self.old_col] = self.df[self.old_col].interpolate(method='linear')

    def _proportion(self, mask):
        ratio = self.df[self.new_col].dropna() / self.df[self.old_col].dropna()
        self.df.loc[mask, self.new_col] = np.mean(ratio) * self.df.loc[mask, self.old_col]

    def _linear_regression(self, mask):
        valid = self.df.dropna(subset=[self.new_col, self.old_col])
        if valid.empty:
            raise ValueError("Not enough data for linear regression.")

        X = valid[self.old_col].values.reshape(-1, 1)
        y = valid[self.new_col].values
        model = LinearRegression().fit(X, y)
        self.df.loc[mask, self.new_col] = model.predict(self.df.loc[mask, self.old_col].values.reshape(-1, 1))

    def _polynomial_regression(self, mask):
        valid = self.df.dropna(subset=[self.new_col, self.old_col])
        if valid.empty:
            raise ValueError("Not enough data for polynomial regression.")

        X = valid[self.old_col].values.reshape(-1, 1)
        y = valid[self.new_col].values
        pipe = make_pipeline(PolynomialFeatures(), LinearRegression())
        grid = GridSearchCV(pipe, {'polynomialfeatures__degree': np.arange(1, 6)}, cv=5)
        grid.fit(X, y)
        self.df.loc[mask, self.new_col] = grid.predict(self.df.loc[mask, self.old_col].values.reshape(-1, 1))

    def _exponential_smoothing(self, mask, alpha=0.5):
        values = self.df[self.new_col].dropna().ewm(alpha=alpha).mean()
        self.df.loc[mask, self.new_col] = values.iloc[-1]

    def _mlp_regression(self, mask):
        valid = self.df.dropna(subset=[self.new_col, self.old_col])
        if valid.empty:
            raise ValueError("Not enough data for MLP regression.")

        X = valid[self.old_col].values.reshape(-1, 1)
        y = valid[self.new_col].values
        model = MLPRegressor(hidden_layer_sizes=(1000,), max_iter=10000, activation="tanh", alpha=0.001, random_state=0)
        model.fit(X, y)
        self.df.loc[mask, self.new_col] = model.predict(self.df.loc[mask, self.old_col].values.reshape(-1, 1))

    def retropolarize(self, method='proportion'):
        mask = self.df[self.new_col].isna() & self.df[self.old_col].notna()

        if method == 'proportion':
            self._proportion(mask)
        elif method == 'linear_regression':
            self._linear_regression(mask)
        elif method == 'polynomial_regression':
            self._polynomial_regression(mask)
        elif method == 'exponential_smoothing':
            self._exponential_smoothing(mask)
        elif method == 'mlp_regression':
            self._mlp_regression(mask)
        else:
            raise ValueError(f"Unknown method: {method}")

        return self.df[self.new_col]

from scrapper.extract_xlsx import Portfolio
import pandas as pd
import numpy as np
from markowitz.OptimizedPortfolio import assert_right_type
from functools import reduce


class NavSeries:
    """Computes statistics over Equity Curves."""

    @staticmethod
    def load(df: pd.DataFrame):
        assert_right_type(df, pd.DataFrame)
        return NavSeries(navs=df)

    def __init__(self, df: pd.DataFrame = None, rf: float = 0., periods: int = None, **kwargs):
        self.__navs = df or reduce(lambda x, y: pd.merge(left=x, right=y, on='date'), kwargs.values())
        self.__periods_per_year: int = 12 or periods
        self.__rf: float = rf

    def get_navs(self):
        return self.__navs

    def summary(self, periods: int = None):
        periods: int = periods or self.__periods_per_year
        di = dict()

        di["Annual Return"] = 100 * self.__navs.pct_change().dropna().mean().apply(lambda x: (1 + x) ** periods - 1)
        di["Annual Volatility"] = 100 * self.__navs.pct_change().dropna().std() * np.sqrt(periods)
        di["Annual Sharpe Ratio"] = di["Annual Return"] / di["Annual Volatility"]
        return pd.DataFrame.from_dict(di, orient='index').transpose()

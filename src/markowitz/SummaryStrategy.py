import pandas as pd
import numpy as np
from utils import assert_right_type


class SummaryStrategy:
    def __init__(self, portfolio_returns: pd.DataFrame, periods: int = None, logreturns: bool = True):
        self.__periods: int = 12 or periods
        assert_right_type(portfolio_returns, pd.DataFrame)
        assert_right_type(logreturns, bool)
        self.__returns: pd.DataFrame = portfolio_returns
        self.__sum_stats: pd.DataFrame = pd.DataFrame()
        self.__is_log: bool = logreturns

    def compute(self):
        m: int = self.__periods
        self.__sum_stats["Annual Mean"] = self.__returns.mean() * m * 100 if self.__is_log \
            else self.__returns.mean().add(1).apply(lambda x: x ** m).sub(1) * 100
        self.__sum_stats["Annual Std"] = self.__returns.std() * np.sqrt(self.__periods) * 100
        self.__sum_stats["Sharpe Ratio"] = self.__sum_stats["Annual Mean"] / self.__sum_stats["Annual Std"]
        self.__sum_stats["Skewness"] = (self.__returns.mul(self.__periods * 100)).skew()
        return self.__sum_stats


if __name__ == '__main__':
    pass

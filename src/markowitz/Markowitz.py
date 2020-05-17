import numpy as np
from utils import assert_right_type
from sklearn.linear_model import LinearRegression
from copy import deepcopy


class Markowitz:
    def __init__(self, data: np.ndarray, rf: np.ndarray = None, logreturns: bool = True, gamma: float = 3.):
        """
        Compute mean-variance optimization with sample covariance matrix for the case of N risky assets.
        The risk-free rate 'rf' is utilized just for the optimization of excess returns.

        :param data: T-by-N price matrix with assets on axis=1 and time series on axis=0.
        :param rf: Risk-free rate. If the purpose is to optimize excess returns, then provide rf.
        :param logreturns: Log-return if True, else simple return.
        """
        assert_right_type(data, np.ndarray)
        self.__data: np.ndarray = data  # (T, N)
        # log_prices: np.ndarray = np.log1p(data)  # (T, N)
        # self.__log_returns: np.ndarray = np.diff(log_prices, axis=0)  # (T, N-1)
        return_type: dict = {True: Markowitz._log_returns, False: Markowitz._simple_returns}

        assert_right_type(gamma, float)
        self.__gamma = gamma
        self.__t: int = data.shape[0]
        self.__n: int = data.shape[1]
        if isinstance(rf, np.ndarray):
            assert_right_type(rf, np.ndarray)
            self.__rf: np.ndarray = rf  # (T-1, 1)
        else:  # rf = None
            self.__rf: np.ndarray = np.zeros((self.__t - 1, 1))  # (T-1, 1)

        # Log or simple returns; in excess of the risk-free rate if passed as argument.
        self.__returns: np.ndarray = return_type[logreturns](self.__data, self.__rf)  # (T-1, N)

        # Expected Returns
        self.__mu: np.ndarray = np.apply_along_axis(func1d=np.mean, axis=0, arr=self.__returns)  # shape -> (N,)

        # Sample covariance matrix
        self._sigma = np.cov(self.__returns.T)  # shape -> (N, N)

    @staticmethod
    def _log_returns(prices: np.ndarray, rf: np.ndarray) -> np.ndarray:
        r = np.diff(np.log1p(prices), axis=0) - rf
        # Check if there is an 'inf' left in the matrix
        assert not np.isinf(r).any(), f"'inf' values left in returns."
        return r  # (T-1, N)

    @staticmethod
    def _simple_returns(prices: np.ndarray, rf: np.ndarray) -> np.ndarray:
        # np.nan prevent 'inf' values to happen most of the time, at the end we convert them to 0
        r = np.diff(prices, axis=0) / prices[:-1, :] - rf
        assert not np.isinf(r).any(), f"'inf' values left in returns."
        return r  # (T-1, N)

    def _sigma_inv(self) -> np.ndarray:
        return np.linalg.inv(self._sigma)  # shape -> (N, N)

    def compute(self) -> np.ndarray:
        w_star = (1 / self.__gamma) * np.matmul(self._sigma_inv(), self.__mu)
        return deepcopy(w_star)  # shape -> (N,)

    def data(self) -> np.ndarray:
        return deepcopy(self.__data)

    def returns(self):
        return deepcopy(self.__returns)

    # def log_returns(self) -> np.ndarray:
    #     return self.__log_returns

    def mu(self) -> np.ndarray:
        return deepcopy(self.__mu)


class MarkowitzShrinkedConstantCorrelation(Markowitz):  # Constant Correlation
    def __constant_correlation(self):
        cor_matrix: np.ndarray = np.corrcoef(self.returns().T)
        mean_cor: np.ndarray = np.mean(cor_matrix)
        rows, columns = cor_matrix.shape
        f: np.ndarray = np.zeros((rows, columns))
        # Far from being optimized
        for i in range(rows):
            for j in range(columns):
                f[i][j] = mean_cor * np.sqrt(cor_matrix[i][i] * cor_matrix[j][j])
        return f

    def _sigma_inv(self) -> np.ndarray:
        lambda_ = .2
        f = self.__constant_correlation()
        shrinked = f * lambda_ + (1 - lambda_) * self._sigma
        return np.linalg.pinv(shrinked)


class MarkowitzShrinkedIdentity(Markowitz):  # Identity Matrix
    def _sigma_inv(self) -> np.ndarray:
        lambda_ = .2
        f = np.identity(self._sigma.shape[0])
        shrinked = f * lambda_ + (1 - lambda_) * self._sigma
        return np.linalg.pinv(shrinked)


class MarkowitzShrinkedSI(Markowitz):  # Single Index
    def __market_portfolio(self):
        data = self.data()
        log = np.log1p(data)
        diff = np.diff(log, axis=0)
        market = np.apply_along_axis(np.sum, 1, diff)
        return market

    def __f(self):
        market = self.__market_portfolio()
        returns = self.returns()
        reg = LinearRegression().fit(returns, market)
        pred = reg.predict(returns)
        residuals = market - pred
        residuals_variance = np.fromiter(
            (np.cov(returns[:, i], residuals)[0, 1] for i in range(returns.shape[1])),
            dtype=float)
        D = np.identity(residuals_variance.shape[0]) * residuals_variance
        s_2 = market.var()
        beta = reg.coef_
        return s_2 * beta * beta.reshape((-1, 1)) + D

    def _sigma_inv(self) -> np.ndarray:
        lambda_ = .2
        shrinked = self.__f() * lambda_ + (1 - lambda_) * self._sigma
        return np.linalg.pinv(shrinked)

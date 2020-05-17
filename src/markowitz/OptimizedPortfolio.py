import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import markowitz
from scrapper.extract_xlsx import Portfolio
from pypfopt.efficient_frontier import EfficientFrontier
from utils import is_multiple_of, pair, assert_right_type, err_msg
from copy import deepcopy

plt.style.use("seaborn-dark")
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # bluish dark grey
for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey


class OptimizedPortfolio(Portfolio):

    def __init__(self, prices: pd.DataFrame = None,
                 mv: pd.DataFrame = None,
                 rf: pd.Series or pd.DataFrame = None,
                 excess: bool = True,
                 rolling_window: int = 120,
                 scheme: str = 'mean_variance',
                 cov_estimation_method: str = 'sample',
                 gamma: float = 3.,
                 logreturns: bool = True,
                 verbose: bool = False,
                 visualize_backtest: bool = True):
        super().__init__(prices, mv, rf, logreturns)
        """
        Optimizes a portfolio according to a weighting scheme strategy.

        :param portfolio:
        :param excess: if we deal with excess returns or not (default: True).
        :param rolling_window: a time rolling window.
        :param scheme: The weighting scheme. Can be:
            - 'mean_variance',
            - 'market_cap_weighted',
            - 'equally_weighted'.
        :param cov_estimation_method: How to estimate the covariance matrix:
            - 'sample': sample covariance matrix.
            - 'shrink_1FFF': sample covariance matrix shrunk towards the single-index covariance matrix
            - 'shrink_3FFF'
            - 'shrink_5FFF'
        :param gamma: The risk aversion parameter.
        """

        assert_right_type(excess, bool)
        assert_right_type(visualize_backtest, bool)
        self.__excess: bool = excess
        assert_right_type(rolling_window, int)
        assert 0.50 > rolling_window / self.get_t(), \
            f"The rolling window is too big with respect to your data subset! " \
            f"It represents the {int(rolling_window / self.get_t() * 100)}% of your data subset."
        self.__rolling_window: int = rolling_window
        assert_right_type(scheme, str)
        assert_right_type(cov_estimation_method, str)
        assert_right_type(gamma, float)

        # Verify if the weighting scheme passed as argument exists
        self.__schemes = np.array(['mean_variance', 'market_cap_weighted', 'equally_weighted', 'min_variance'])

        assert scheme in self.__schemes, err_msg(scheme, 'weighting scheme', list(self.__schemes))
        self.__weighting_scheme: str = scheme

        self.__weighting_schemes_labels = {'mean_variance': 'MVO', 'market_cap_weighted': 'MC',
                                           'equally_weighted': 'EW', 'min_variance': 'MinVO'}

        self.__weighting_schemes_dict = {'mean_variance':
                                         {'sample': markowitz.Markowitz,
                                          'single_index': markowitz.MarkowitzShrinkedSI,
                                          'identity': markowitz.MarkowitzShrinkedIdentity,
                                          'constant_correlation': markowitz.MarkowitzShrinkedConstantCorrelation}}

        self.__prediction_methods = {'equally_weighted': {'sample': self.__predict_ew},
                                     'market_cap_weighted': {'sample': self.__predict_mc},
                                     'mean_variance': {'sample': self.__predict_meanvar,
                                                       'single_index': self.__predict_meanvar,
                                                       'identity': self.__predict_meanvar,
                                                       'constant_correlation': self.__predict_meanvar}}

        self.__gamma = gamma

        self.__cov_estimation_methods = np.array(['sample', 'constant_correlation',
                                                  'single_index', 'identity'])
        assert cov_estimation_method in self.__cov_estimation_methods, \
            err_msg(cov_estimation_method, 'covariance estimation', list(self.__cov_estimation_methods))
        self.__cov_estimation_method: str = cov_estimation_method

        # Initializing this avoids computing the weights again and again
        # self.__weights: pd.DataFrame = self.__compute_weights(verbose=verbose)
        self.__weights: pd.DataFrame = self.backtest(verbose=verbose, visualization=visualize_backtest)

    def __predict_ew(self, start: np.datetime64, end: np.datetime64) -> pd.DataFrame:
        # Prediction for excess returns or returns
        is_excess: bool = self.__excess

        # Get the simple/log-returns subset (in excess of the risk-free rate or not)
        returns_subset: pd.DataFrame = self.excess_returns(start, end) if is_excess else self.returns(start, end)

        # Get the number of stocks
        numstocks: int = len(returns_subset.columns)

        # Create the equally-weighted subset DataFrame of weights; weight t+1 "predicted" at t
        w_t1: pd.DataFrame = pd.DataFrame(np.repeat(1 / numstocks, numstocks).reshape((1, numstocks)),
                                          index=[end], columns=returns_subset.columns)  # shape -> (1, N)

        return w_t1

    def __predict_mc(self, start: np.datetime64, end: np.datetime64) -> pd.DataFrame:
        # Prediction for excess returns or returns
        is_excess: bool = self.__excess

        # Get the simple returns subset
        returns_subset: pd.DataFrame = self.excess_returns(start, end) if is_excess else self.returns(start, end)

        # Get the market values subset
        mv_subset: pd.DataFrame = self.create_window(returns_subset.index[0], end, key='mv')

        # Modify windows so that both (returns and market values) have the same shape
        if mv_subset.shape[1] > returns_subset.shape[1]:
            mv_subset = mv_subset.loc[:, returns_subset.columns]
        elif mv_subset.shape[1] < returns_subset.shape[1]:
            mv_subset = returns_subset.loc[:, mv_subset.columns]

        # Verify if both windows (returns and market values) have the same shape
        assert mv_subset.shape == returns_subset.shape, f"'mv.shape': {mv_subset.shape} does not match " \
                                                        f"'returns_subset.shape': {returns_subset.shape}."

        # Compute the window market value mean
        mv_subset: pd.Series = mv_subset.mean(axis=0)  # shape -> (N,)

        # Market cap weights
        w_t1: pd.DataFrame = mv_subset.apply(lambda weight: weight / mv_subset.sum()).to_frame().transpose()
        w_t1.index = [end]

        return w_t1  # shape -> (1, N)

    def __predict_meanvar(self, start: np.datetime64, end: np.datetime64) -> pd.DataFrame:
        # Get the prices subset
        prices_subset: pd.DataFrame = self.create_window(start, end, 'p')

        is_log: bool = self.get_islog()
        is_excess: bool = self.__excess
        rf: pd.DataFrame = self.create_window(start, end, 'rf').iloc[1:].to_numpy() if is_excess else None

        # Covariance estimation method
        cov_est_method: str = self.__cov_estimation_method

        # Optimization program
        opt_program: str = self.__weighting_scheme

        # Compute the right optimization with right covariance estimation
        kwargs = dict(data=prices_subset.to_numpy(), rf=rf, logreturns=is_log, gamma=self.__gamma)
        opt_object = self.__weighting_schemes_dict[opt_program].get(cov_est_method)(**kwargs)

        # Compute the mean-variance weights; shape -> (N,)
        optimal_w_t1: np.ndarray = opt_object.compute()

        # Initialize the subset's weights DataFrame; weight t+1 predicted at t; shape -> (1, N)
        w_t1 = pd.DataFrame(optimal_w_t1.reshape((1, len(optimal_w_t1))), index=[end],
                            columns=prices_subset.columns)
        return w_t1

    @staticmethod
    def verify_constraints(w_t: pd.DataFrame):
        # Make sure that short selling is prohibited
        w_t = w_t.clip(0.)  # shape -> (1, N)
        w_sum: pd.DataFrame = w_t.sum(axis=1)  # shape -> (1,)

        # Normalize weights if their sum is greater than 1
        if w_sum.values > 1:
            w_t = np.true_divide(w_t, w_sum.values)
        return w_t  # shape -> (1, N)

    def predict_weights(self, start: np.datetime64, end: np.datetime64):
        """Predict optimal weights according to the right optimization program."""
        # Get the prices subset
        # prices_window: pd.DataFrame = self.create_window(start, end, 'p')

        # Covariance estimation method
        cov_est_method: str = self.__cov_estimation_method

        # Optimization program
        opt_program: str = self.__weighting_scheme

        # Compute weights at time t with the right optimization with right covariance estimation
        # w_t: pd.DataFrame = self.__prediction_methods[opt_program].get(cov_est_method)(prices_window)
        w_t: pd.DataFrame = self.__prediction_methods[opt_program].get(cov_est_method)(start, end)
        # E.g. self.__prediction_methods['equally_weighted'].get('sample')(start, end)
        return w_t

    def backtest(self, verbose: bool = True, visualization: bool = True):
        """Backtest that computes weights over rolling windows."""
        # Verify that the company names array length is equal to the number of companies in the dataset
        assert len(self.get_company_names()) == self.get_n(), \
            f"The number of stocks does not match: {len(self.get_company_names())} stock names " \
            f"vs. {self.get_n()} stocks."

        # Get the boolean indicating whether we are dealing with excess or normal returns
        is_excess: bool = self.__excess

        # Get the boolean indicating whether we are dealing with log-returns or not
        is_log: bool = self.get_islog()

        # Get the optimization program name
        program_name: str = self.__weighting_scheme

        # Get the number of companies
        numstocks: int = self.get_n()

        # Get the entire number of periods of the dataset
        numperiods: int = self.get_t()

        # Get the number of periods of the rolling window
        rollwin: int = self.__rolling_window

        # Retrieve the dates
        dates: np.ndarray = self.get_dates()

        # Get the whole prices dataset; dimensions -> (T, N)
        prices: pd.DataFrame = self.get_prices()

        # Initialize the big weights' matrix; dimensions -> (T - rolling window, N)
        # Setup of the DataFrame of weights over the whole rollover periods to harvest the optimal weights
        w: pd.DataFrame = pd.DataFrame(data=np.tile(0., (numperiods - rollwin, numstocks)),
                                       index=dates[rollwin:],
                                       columns=self.get_company_names())

        # Create the windows to roll over
        windows: list = [(t, t + rollwin) for t in range(numperiods - rollwin)]

        # Get the whole return history; dimensions -> (T-1, N)
        normal_returns: pd.DataFrame = self.log_returns(prices) if is_log else self.simple_returns(prices)
        # normal_returns: pd.DataFrame = np.log1p(self.get_prices()).diff(axis=0).dropna(how='all') if is_log \
        #     else self.get_prices().pct_change().dropna(how='all')  # (T-1, N)

        # We retrieve the risk-free rate and slice it to match the normal_returns DataFrame.
        rf: np.ndarray = self.get_rf().iloc[1:].to_numpy()  # (T-1, 1)

        # We only work with simple returns as it is contemporaneous aggregation; dimensions -> (T-1, N)
        r_full: pd.DataFrame = normal_returns.sub(rf, axis='columns') if is_excess else normal_returns
        # r_full_shifted = r_full.tshift(-1).dropna(how='all')

        # Retrieve the (latest) returns that will allow us to compute portfolio returns in live;
        # dimensions -> (T - rollwin - 1, N)
        # r: pd.DataFrame = r_full.iloc[rollwin - 1:, :].fillna(0)
        r: pd.DataFrame = r_full.iloc[rollwin:, :].fillna(0)

        # The shape of the weights' matrix should be (T - rollwin, N) and the shape of the returns
        # should be (T - rollwin - 1, N). The weights' matrix has one more observation than the returns'
        # one since weights are predictions for each next periods. Meaning, that the last weights' row
        # should be used to compute returns that have not happen yet.
        assert w.iloc[1:, :].shape == r.shape, f"Shapes do not match between weights and returns." \
                                               f" w={w.iloc[1:, :].shape} versus r={r.shape}"

        if verbose:
            print(f"\tIteration over {len(windows)} windows:", end="\n\t")

        x, y, z = [], np.array([]), np.array([])

        colors = [
            '#08F7FE',  # teal/cyan
            '#FE53BB',  # pink
            '#a6e22e',  # apple green
            '#F5D300',  # yellow
        ]

        if visualization:
            fig, ax = plt.subplots(1, 1, figsize=(9, 5))
            fig.suptitle(f"{self.__weighting_schemes_labels[program_name]} "
                         f"{rollwin}-months rolling window "
                         f"{'excess' if is_excess else 'normal'} "
                         f"{'log' if is_log else 'simple'} returns ({self.__cov_estimation_method} cov)",
                         fontweight='bold')
            plt.xticks(rotation=45)

        for i, window in enumerate(windows):
            if verbose:
                char = "." if pair(i + 1) else "|"
                end_str = f"{i + 1}th iteration...\n\t" if (is_multiple_of(i + 1, 50)) else ""
                print(f"{char}", end=end_str)

            start, end = dates[window[0]], dates[window[1]]

            # Compute next period weights; dimension -> (1, x), where x a number in range [0, N]
            w_t1: pd.DataFrame = self.predict_weights(start, end)

            # Avoid short selling and normalize
            w_t1_constrained: pd.DataFrame = self.verify_constraints(w_t1)

            # Plug weights in the matrix of weights
            w.loc[w_t1.index, w_t1.columns] = w_t1_constrained

            if visualization and (i < len(windows) - 1):
                t1: np.datetime64 = dates[windows[i + 1][1]]
                ptf_r: np.ndarray = w.tshift(1).loc[t1, :].to_numpy() @ r.loc[t1, :].to_numpy().T

                # Add the portfolio return to z
                z: np.ndarray = np.append(z, float(ptf_r))

                # Compute the cumulative returns
                cum_r: np.ndarray = np.cumsum(z) if is_log else np.cumprod(1 + z) - 1

                # Update the y-axis array
                y: np.ndarray = cum_r
                # y = w.mul(r).sum(axis=1).add(1).cumprod().sub(1)

                # Append the date t to x-axis array
                x.append(t1)
                # x.append(end)

                # Plot the current state of the NAV backtest
                y_percent = np.multiply(y, 100)
                ax.cla()
                ax.plot(x, y_percent, linewidth=1.5, color=colors[0])
                ax.fill_between(x, y_percent, alpha=0.1, color=colors[0])
                ax.plot(x, pd.DataFrame(y_percent).expanding().min(), color=colors[1], linewidth=1)
                ax.plot(x, pd.DataFrame(y_percent).expanding().max(), color=colors[2], linewidth=1)
                ax.set_ylabel('%')
                ax.set_xlabel('Date')
                ax.grid(color='#2A3459', ls='-.')
                plt.tight_layout()
                fig.canvas.draw()

        if visualization:
            plt.show()

        return w

    @staticmethod
    def __portfolio_returns(w_shifted: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
        return r.mul(w_shifted).sum(axis=1)

    def portfolio_returns(self) -> pd.DataFrame:
        # Shift the weights one period to prevent hindsight bias
        w_star_shifted = self.weights().tshift(1).iloc[:-1, :]

        # Prediction for excess returns or returns
        is_excess: bool = self.__excess

        # Retrieve returns using w_star first's date, not w_star_shifted's as computing returns
        # costs one observation
        r: pd.DataFrame = self.excess_returns(adaptive=False) if is_excess else self.returns(adaptive=False)

        # The number of columns must match (forcing)
        if w_star_shifted.shape[1] > r.shape[1]:
            w_star_shifted = w_star_shifted.loc[:, r.columns]
        elif w_star_shifted.shape[1] < r.shape[1]:
            r = r.loc[:, w_star_shifted.columns]
        else:
            pass

        # The indices will not match because the weights are one observation ahead from realized returns
        r = r.loc[w_star_shifted.index, :]

        assert r.shape == w_star_shifted.shape, f"Returns' shape is {r.shape}, while weights' " \
                                                f"shape is {w_star_shifted.shape}. " \
                                                f"They have to match!"

        isexs: str = "_excess" if is_excess else "_normal"

        islogs: str = "_log" if self.get_islog() else "_simple"

        col_name = f'{self.__weighting_scheme}_{self.__rolling_window}_portfolio{isexs}'\
                   f'{islogs}_returns_{self.__cov_estimation_method}'

        # Compute the portfolio weighted return and convert it to a DataFrame
        ptf_r = self.__portfolio_returns(w_star_shifted, r).to_frame(name=col_name)

        # Make sure the index name is 'date'
        ptf_r.index.name = 'date'

        return ptf_r

    def cumulative_portfolio_returns(self) -> pd.DataFrame:
        is_log = self.get_islog()
        cpr: pd.DataFrame = self._cumulative_returns(self.portfolio_returns(), is_log)
        cpr.index.name = 'date'
        return cpr

    def weights(self) -> pd.DataFrame:
        """Warning: These weights come from backtest. Meaning that they are already
        shifted one period ahead as they were computed as being the predictive
        weights for the next period (t+1) but indexed at date t."""
        return deepcopy(self.__weights)

    @staticmethod
    def __min_vol(mu: pd.Series, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        # Setup
        ef = EfficientFrontier(mu, cov_matrix)

        # Optimizes for minimum volatility
        min_ptf = ef.min_volatility()
        return pd.DataFrame.from_dict(data=min_ptf, orient='index').T

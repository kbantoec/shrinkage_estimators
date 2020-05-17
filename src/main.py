import numpy as np
from scrapper import DataLoader, Portfolio
from markowitz import Markowitz, MarkowitzShrinkedConstantCorrelation, MarkowitzShrinkedIdentity, MarkowitzShrinkedSI
from markowitz import OptimizedPortfolio
from markowitz.SummaryStrategy import SummaryStrategy
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from utils import import_data, assert_right_type
from backtest_results import *
import copy


def main():
    p, mv, rf = import_data()

    data = DataLoader(prices=p, mv=mv, rf=rf)

    prices_pd = data.create_window('2003', '2005', 'p')
    prices = prices_pd.to_numpy()
    markowitz = Markowitz(prices)
    w = markowitz.compute()
    w_returns = np.apply_along_axis(np.sum, 0, (w * markowitz.returns()))
    markowitz_shrinked = MarkowitzShrinkedConstantCorrelation(prices)
    w_shrinked = markowitz_shrinked.compute()
    w_shrinked_returns = np.apply_along_axis(np.sum, 0,
                                             (w_shrinked * markowitz_shrinked.returns()))
    plt.plot([i for i in range(w_returns.shape[0])], np.cumsum(w_returns) * 100, label="Markowitz")
    plt.plot([i for i in range(w_shrinked_returns.shape[0])], np.cumsum(w_shrinked_returns) * 100,
             label="Markowitz Shrinked")
    plt.xlabel('data')
    plt.ylabel('%')
    plt.title('Markowitz vs Shrinked')
    plt.legend()
    plt.show()

    portfolios = import_portfolios()
    sum_strat = SummaryStrategy(portfolios["porfolio_excess_returns"]).compute()
    print(sum_strat)


def backtester(excess: tuple = (True, False), rolling_window: tuple = (30, 120), programs: list = None,
               cov_estimation_method: tuple = ('sample',), logreturns: bool = True,
               save: bool = False, verbose: bool = True, visualize_backtest: bool = True,
               dirpaths: dict = None) -> dict:
    """
    Backtest of all covariance estimation methods for each program passed as argument.

    :param visualize_backtest: To display live plot of the NAV curve resulting of the backtest.
    :param excess: Specify excess or normal returns.
    :param rolling_window: Number of perdiods for the rolling window.
    :param programs: Optimization programs.
    :param cov_estimation_method: Covariance estimation method (to be developed further).
    :param logreturns: Specify if log-returns or simple returns.
    :param save: Option that allows to save the backtested output data (portfolio returns,
                 cumulative portfolio returns, optimal weights).
    :param verbose: To display the progression of the backtest.
    """

    # Verify that the arguments are of the right type
    assert_right_type(logreturns, bool)
    assert_right_type(verbose, bool)
    assert_right_type(save, bool)
    assert_right_type(excess, tuple)
    assert_right_type(rolling_window, tuple)

    # Import the prices, market values, and risk-free rate data
    p, mv, rf = import_data()

    # If 'programs' is set to 'None', then by default run the backtest on mean-variance optimization
    programs: list = ['mean_variance'] if programs is None else programs
    assert_right_type(programs, list)

    # if 'paths' is not specified
    dirpaths: dict = dict(navs_dir='../data/navs/', rets_dir='../data/rets/',
                          weights_dir='../data/weights/') if dirpaths is None else dirpaths

    programs_str: str = '_'.join(programs)
    logreturns_str: str = "log1p" if logreturns else "simple"
    progs = {'mean_variance': {'cov_estimation_method': ('sample',
                                                         'constant_correlation',
                                                         'single_index',
                                                         'identity')},
             'equally_weighted': {'cov_estimation_method': ('sample',)},
             'market_cap_weighted': {'cov_estimation_method': ('sample',)},
             'min_variance': {'cov_estimation_method': ('sample',)}}

    # Check that there is no wrong optimization program passed
    wrong_programs = [pr for pr in programs if pr not in progs.keys()]
    assert len(wrong_programs) == 0, f"Wrong program passed: {', '.join(wrong_programs)} don't exist."

    chosen_progs: dict = {prog: progs[prog] for prog in programs if prog in progs.keys()}

    counter: int = 0
    for v in chosen_progs.values():
        counter += len(v['cov_estimation_method'])

    # Number of optimizations to compute
    numopts: int = len(excess) * len(rolling_window) * counter

    # Parameters to iterate over
    programs_params: dict = dict(excess=excess, rolling_window=rolling_window, programs=chosen_progs)

    # Initialize an empty dictionnary to store the result of the backtest
    portfolios_dict: dict = dict()

    cov_est_li: list = []

    iteration = 1
    for rollwin in programs_params['rolling_window']:
        for is_excess in programs_params['excess']:
            return_name = 'excess' if is_excess else 'normal'
            for program, tup in programs_params['programs'].items():
                for cov_est in tup['cov_estimation_method']:
                    cov_est_li.append(cov_est)
                    start = time.time() if verbose else None
                    if verbose:
                        print(f"\n{'*' * 80}")
                        print(f"Starting optimization for {program.upper()} program"
                              f"\n\tRolling window: {rollwin},"
                              f"\n\tExcess return: {is_excess},"
                              f"\n\tType of returns: {logreturns_str},"
                              f"\n\tCovariance Matrix Estimation Method: {cov_est}.")

                    ptf = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(),
                                             excess=is_excess,
                                             rolling_window=rollwin,
                                             scheme=program,
                                             logreturns=logreturns,
                                             cov_estimation_method=cov_est,
                                             verbose=verbose,
                                             visualize_backtest=visualize_backtest)

                    # Affect the OptimizedPortfolio instance
                    name_li = [program, rollwin, return_name, cov_est]
                    portfolios_dict[f"{'_'.join(map(str, name_li))}"] = ptf

                    if verbose:
                        name = f"[{' '.join(map(str, name_li)).upper()}]"
                        t = time.time() - start
                        t_str = f"{t / 60:.2f} minutes" if t >= 60 else f"{t:.2f} seconds"
                        print(f"\nEllapsed time for {name}: {t_str}.")
                        print(f"{iteration}/{numopts} optimizations completed.")
                        iteration += 1

    # Extract the NAVs and export to feather format
    if save:
        # Collect portfolio returns in a list
        porfolio_returns: list = [portfolios_dict[k].portfolio_returns() for k in portfolios_dict.keys()]

        # Add the cumulative portfolio returns to a list
        cumrets: list = [portfolios_dict[k].cumulative_portfolio_returns() for k in portfolios_dict.keys()]

        # Add the optimal weights to a list
        optimal_weights: list = [portfolios_dict[k].weights() for k in portfolios_dict.keys()]

        # Concatenate the list of DataFrames into a single DataFrame
        optimized_portfolios: pd.DataFrame = pd.concat(cumrets, axis=1)

        # Get the labels of each portfolio
        portfolio_names: list = optimized_portfolios.columns

        # Convert parameters to string
        excess_params: str = "_".join(['excess' if is_excess else 'normal' for is_excess in excess])
        rolling_window_params: str = '_'.join(map(str, rolling_window))

        # Drop duplicates from the list
        cov_est_str: str = "_".join(list(dict.fromkeys(cov_est_li)))

        # Generic filename
        filename: str = f"{programs_str}_{rolling_window_params}_{excess_params}_{logreturns_str}_" \
                        f"{cov_est_str}.feather"

        # Save portfolio returns
        for r, col in zip(porfolio_returns, portfolio_names):
            r.reset_index().rename({'index': 'date'}, axis=1).to_feather(f"{dirpaths['rets_dir']}{col}.feather")
            print(f"Portfolio returns saved as: '{dirpaths['rets_dir']}{col}.feather'")

        # Save cumulative portfolio returns
        cumrets_path = f"{dirpaths['navs_dir']}{filename}"
        optimized_portfolios.reset_index().rename({'index': 'date'}, axis=1).to_feather(cumrets_path)
        print(f"NAV curves saved as: {cumrets_path!r}")

        # Save weights
        for w, col in zip(optimal_weights, portfolio_names):
            w.reset_index().rename({'index': 'date'}, axis=1).to_feather(f"{dirpaths['weights_dir']}{col}.feather")
            print(f"Weights saved as: '{dirpaths['weights_dir']}{col}.feather'")

    return portfolios_dict


if __name__ == "__main__":

    # Parameter specifications
    main_dir: str = 'C:/Users/YBant/Desktop/backtests_1705/'

    # To plot monthly GS1M vs. F-F RF (their 1-month TBill return is from Ibbotson and Associates)
    # plot_rf_vs_ff(main_dir)

    dirpaths: dict = dict(navs_dir=f'{main_dir}data/navs/',
                          rets_dir=f'{main_dir}data/portfolio_returns/',
                          weights_dir=f'{main_dir}data/weights/')

    linewidth = 1

    palette = dict(mvo_sample='#009432', mvo_cc='#5352ed', mvo_si='#e056fd',
                   mvo_id='#8e44ad', ew='#ff4757', mc='#ffa502')

    # kwargs: dict = dict(excess=(True,), rolling_window=(120,), programs=['equally_weighted'],
    #                     logreturns=False, save=False, verbose=True, visualize_backtest=True,
    #                     dirpaths=dirpaths)
    #
    # result: dict = backtester(**kwargs)
    # obj = result['equally_weighted_120_excess_sample']
    # cr = obj.cumulative_portfolio_returns()
    # plt.plot(cr.mul(100))
    # plt.show()

    # optw = obj.weights().tshift(1).iloc[:-1, :]
    # # prices.pct_change().dropna(axis=0, how='all')
    # er = obj.get_prices().pct_change().dropna(axis=0, how='all').sub(obj.get_rf()[1:].to_numpy(), axis='columns')
    # er = er.loc[optw.index, :]
    # print(optw.shape, er.shape)
    # ptfr = er.mul(optw).sum(axis=1)
    # # ptfr = optw.mul(er).sum(axis=1)
    # cumr = ptfr.add(1).cumprod().sub(1).mul(100)
    # # cumr = ptfr.cumsum().mul(100)
    # # cumr = obj.portfolio_returns().add(1).cumprod().sub(1).mul(100)
    # plt.plot(cumr)
    # plt.show()

    kwargs = dict(main_dir=main_dir, dirpaths=dirpaths, cmap=palette, linewidth=linewidth)

    # BACKTESTS
    #############
    # ew_120en = backtest_excess_vs_normal_navs(**kwargs, color='ew', program='equally_weighted',
    #                                           rolling_window=120, save=True, visualize_backtest=True)
    # ew_30en = backtest_excess_vs_normal_navs(**kwargs, color='ew', program='equally_weighted',
    #                                         rolling_window=30, save=True, visualize_backtest=True)
    # all_120e = backtest_all(**kwargs, save=True)
    # all_90e = backtest_all(**kwargs, rolling_window=90, save=True, visualize_backtest=False)
    # all_60e = backtest_all(**kwargs, rolling_window=60, save=True, visualize_backtest=False)
    # all_50e = backtest_all(**kwargs, rolling_window=50, save=True, visualize_backtest=False)
    # all_40e = backtest_all(**kwargs, rolling_window=40, save=True, visualize_backtest=False)
    # all_30e = backtest_all(**kwargs, rolling_window=30, save=True, visualize_backtest=False)

    # all_120n = backtest_all(**kwargs, excess=False, save=True, visualize_backtest=False)
    # all_30n = backtest_all(**kwargs, excess=False, rolling_window=30, save=True)

    # Plot of the Excess cumulative portfolio simple returns for a 30-month rolling window
    plt.style.use('classic')
    cumer30 = pd.read_feather(f"{main_dir}data/navs/cumer30.feather").set_index('date').mul(100)
    labels = ['Mean Variance (Single-Index)', 'Mean Variance (Constant Correlation)',
              'Mean Variance (Sample)', 'Mean Variance (Identity)',
              'Equally-Weighted', 'Market-Cap Weighted']
    colors = ['#e056fd', '#5352ed', '#009432', '#8e44ad', '#ff4757', '#ffa502']
    plt.figure(figsize=(9, 5), facecolor='w')
    plt.title("Excess cumulative portfolio simple returns for a 30-month rolling window")
    for i, col in enumerate(cumer30):
        plt.plot(cumer30[col], color=colors[i])
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, ls='dashdot', alpha=0.5)
    plt.legend(labels, loc='best', prop={'size': 12})
    plt.ylabel('%')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig(f"{main_dir}plots/navs_excess_30.png", dpi=600, transparent=True)
    plt.show()

    ptfer30 = pd.read_feather(f"{main_dir}data/portfolio_returns/ptfer30.feather").set_index('date')
    sum_strat30 = SummaryStrategy(ptfer30, logreturns=False).compute()
    print(sum_strat30)
    sum_strat30.to_excel(f"{main_dir}data/sum_stats/ptfer30.xlsx")

    ptfer120file = ['mean_variance_120_portfolio_excess_simple_returns_single_index.feather',
                    'mean_variance_120_portfolio_excess_simple_returns_constant_correlation.feather',
                    'mean_variance_120_portfolio_excess_simple_returns_sample.feather',
                    'mean_variance_120_portfolio_excess_simple_returns_identity.feather',
                    'equally_weighted_120_portfolio_excess_simple_returns_sample.feather',
                    'market_cap_weighted_120_portfolio_excess_simple_returns_sample.feather']
    ptfer120_li = []
    for filename in ptfer120file:
        ptfer120_li.append(pd.read_feather(f"{main_dir}data/portfolio_returns/{filename}").set_index('date'))

    ptfer120 = pd.concat(ptfer120_li, axis=1)
    sum_strat_er120 = SummaryStrategy(ptfer120, logreturns=False).compute()
    print(sum_strat_er120)
    sum_strat_er120.to_excel(f"{main_dir}data/sum_stats/ptfer120.xlsx")


    # # # Import the data
    # p, mv, rf = import_data()
    # data = dict(prices=copy.deepcopy(p), mv=copy.deepcopy(mv), rf=copy.deepcopy(rf))
    #
    # mvo_si = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(),
    #                             excess=True,
    #                             rolling_window=30,
    #                             scheme='mean_variance',
    #                             logreturns=False,
    #                             cov_estimation_method='single_index',
    #                             verbose=True,
    #                             visualize_backtest=False)
    #
    # plt.style.use('classic')
    # cr_si = mvo_si.cumulative_portfolio_returns()
    # plt.plot(cr_si.mul(100))
    # plt.show()

    # optw = mvo_si.weights()
    # numperiods: int = mvo_si.get_t()
    # index_t0: int = numperiods - optw.shape[0] - 2
    # dat = mvo_si.get_dates()

    # r = mvo_si.returns(dat[index_t0])

    # r = mvo_si.returns()
    # if optw.shape[1] > r.shape[1]:
    #     optw = optw.loc[:, r.columns]
    # elif optw.shape[1] < r.shape[1]:
    #     r = r.loc[:, optw.columns]
    # else:
    #     pass
    #
    # r = r.loc[optw.index, :]
    # print(r.shape, optw.shape)  # (214, 400) (214, 400)
    #
    # ptf_rsi = optw.mul(r).sum(axis=1)
    # cum_rsi = ptf_rsi.add(1).cumprod().sub(1)
    # plt.plot(cum_rsi)
    # plt.show()

    # mvo_silr = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(),
    #                               excess=True,
    #                               rolling_window=30,
    #                               scheme='mean_variance',
    #                               logreturns=True,
    #                               cov_estimation_method='single_index',
    #                               verbose=True,
    #                               visualize_backtest=True)



    # mvo_portfolios: dict = backtester(excess=(True,), rolling_window=(30,), programs=['mean_variance'],
    #                                   logreturns=False, save=True, verbose=True, visualize_backtest=False,
    #                                   dirpaths=dirpaths)

    # plt.plot(navs120e['mean_variance_120_portfolio_excess_simple_returns_sample'].mul(100),
    #          label='Mean Variance (Sample)', linewidth=1.5, color='#12CBC4')
    # plt.plot(navs120e['mean_variance_120_portfolio_excess_simple_returns_constant_correlation'].mul(100),
    #          label='Mean Variance (Constant Correlation)', linewidth=1.5, color='#1289A7')
    # plt.plot(navs120e['mean_variance_120_portfolio_excess_simple_returns_single_index'].mul(100),
    #          label='Mean Variance (Single-Index)', linewidth=1.5, color='#0652DD')
    # plt.plot(navs120e['mean_variance_120_portfolio_excess_simple_returns_identity'].mul(100),
    #          label='Mean Variance (Identity)', linewidth=1.5, color='#1B1464')
    # plt.plot(navs120e['equally_weighted_120_portfolio_excess_simple_returns_sample'].mul(100),
    #          label='Equally-Weighted', linewidth=1.5, color='#6F1E51')
    # plt.plot(navs120e['market_cap_weighted_120_portfolio_excess_simple_returns_sample'].mul(100),
    #          label='Market-Cap Weighted', linewidth=1.5, color='#6F1E51')

    # Import the data
    # p, mv, rf = import_data()
    # data = dict(prices=copy.deepcopy(p), mv=copy.deepcopy(mv), rf=copy.deepcopy(rf))
    # prgs = ['mean_variance', 'equally_weighted', 'market_cap_weighted']

    # BACKTESTS
    #############

    # EW
    # kwargs_ = dict(excess=False, rolling_window=120, scheme='equally_weighted', logreturns=False,
    #                cov_estimation_method='sample', verbose=True, visualize_backtest=True)
    # opt120 = OptimizedPortfolio(**data, **kwargs_)
    # print(opt120.cumulative_portfolio_returns().head())
    # plt.title("EW excess log-returns (sample cov)")
    # cr = opt120.cumulative_portfolio_returns()
    # plt.plot(cr)
    # plt.plot(cr.expanding().min())
    # plt.plot(cr.expanding().max())
    # plt.show()

    # meanvar120sim: dict = backtester(excess=(True, ), rolling_window=(120,), programs=['mean_variance'],
    #                                  save=False, logreturns=False, cov_estimation_method=('single_index', ))

    # Combinations to backtest
    # excess: tuple = (True, False)
    # rolling_window: tuple = (30, 120)
    # programs: tuple = ('mean_variance', 'equally_weighted', 'market_cap_weighted')
    # cov_estimation_method: tuple = ('sample',)
    # logreturns: tuple = (True, False)
    #
    # di = dict(excess=excess, rolling_window=rolling_window, programs=programs,
    #           cov_estimation_method=cov_estimation_method, logreturns=logreturns)
    # di.fromkeys(excess, rolling_window)
    # # {True: (30, 120), False: (30, 120)}

    # from itertools import count
    # i = count()
    # x, y = [], []
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.grid(True)
    # while True:
    #     x.append(next(i))
    #     y.append(float(np.random.randn(1)))
    #     ax.plot(x, y)
    #     fig.canvas.draw()
    #     time.sleep(0.5)
    #     if next(i) > 15:
    #         break
    # plt.show()

    # kwargs_ = dict(excess=True, rolling_window=120, scheme='mean_variance', logreturns=False,
    #                cov_estimation_method='single_index', verbose=True, visualize_backtest=True)
    # mv120esrsi = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(), **kwargs_)
    # plt.plot(mv120esrsi.cumulative_portfolio_returns())
    # plt.show()
    #
    # kwargs_ = dict(excess=False, rolling_window=120, scheme='mean_variance', logreturns=False,
    #                cov_estimation_method='single_index', verbose=True, visualize_backtest=True)
    # mv120nsrsi = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(), **kwargs_)
    # plt.plot(mv120nsrsi.cumulative_portfolio_returns())
    # plt.show()

    # kwargs_ = dict(excess=False, rolling_window=120, scheme='mean_variance', logreturns=True,
    #                cov_estimation_method='constant_correlation', verbose=True, visualize_backtest=True)
    # mvo120nlr = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(), **kwargs_)
    # plt.title("EW normal log-returns")
    # plt.plot(mvo120nlr.cumulative_portfolio_returns())
    # plt.show()


    # # EW
    # ######
    # # Log-returns
    # kwargs_ = dict(excess=True, rolling_window=120, scheme='equally_weighted', logreturns=True,
    #                cov_estimation_method='sample', verbose=True, visualize_backtest=True)
    # ew120elr = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(), **kwargs_)
    # plt.title("EW excess log-returns")
    # plt.plot(ew120elr.cumulative_portfolio_returns())
    # plt.show()
    #
    # lr = OptimizedPortfolio.log_returns(ew120elr.get_prices().iloc[120:, :])  # (123, 3303)
    # mu = lr.sum(axis=0)  # (3303,)
    # w = ew120elr.weights()  # (124, 3303)
    # w_t1 = w.shift(1).dropna(axis=0, how='all')  # (123, 3303)
    # ptf_lr = w_t1.dot(mu)
    # cum_lr = np.exp(ptf_lr).cumsum() - 1
    # plt.plot(cum_lr)
    # plt.show()
    #
    #
    # kwargs_ = dict(excess=False, rolling_window=120, scheme='equally_weighted', logreturns=True,
    #                cov_estimation_method='sample', verbose=True, visualize_backtest=True)
    # ew120nlr = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(), **kwargs_)
    # plt.title("EW normal log-returns")
    # plt.plot(ew120nlr.cumulative_portfolio_returns())
    # plt.show()

    # # Simple returns
    # kwargs_ = dict(excess=True, rolling_window=120, scheme='equally_weighted', logreturns=False,
    #                cov_estimation_method='sample', verbose=True, visualize_backtest=True)
    # ew120esr = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(), **kwargs_)
    # kwargs_ = dict(excess=False, rolling_window=120, scheme='equally_weighted', logreturns=False,
    #                cov_estimation_method='sample', verbose=True, visualize_backtest=True)
    # ew120nsr = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(), **kwargs_)
    # plt.title("EW Simple returns")
    # plt.plot(ew120esr.cumulative_portfolio_returns(), label="Excess")
    # plt.plot(ew120nsr.cumulative_portfolio_returns(), label="Normal")
    # plt.legend()
    # plt.show()
    #
    # cr1 = ew120esr.cumulative_portfolio_returns().to_frame("ew_excess_simple_returns")
    # cr1.index.name = 'date'
    # cr2 = ew120nsr.cumulative_portfolio_returns().to_frame("ew_normal_simple_returns")
    # cr2.index.name = 'date'
    #
    # Portfolio.plotter(cr1, cr2)

    # Backtest MCW portfolios
    ###########################
    # mc120sim: dict = backtester(excess=(False, True), rolling_window=(120,), programs=['market_cap_weighted'],
    #                             save=False, logreturns=False)
    #
    # plt.figure(figsize=(9, 5))
    # plt.plot(mc120sim['120_excess_market_cap_weighted_sample'].cumulative_portfolio_returns(),
    #          color='k', label='Excess returns')
    # plt.plot(mc120sim['120_simple_market_cap_weighted_sample'].cumulative_portfolio_returns(),
    #          color='k', label='Normal returns', linestyle=':')
    # plt.legend()
    # plt.tight_layout()
    # plt.grid(True)
    # plt.savefig('C:/Users/YBant/Desktop/backtests_am_project/plots/nav_mkt_cap_weighted_120_excess_vs_normal_sample.png',
    #             transparent=True, dpi=600)
    # plt.show()


    # Backtest EW portfolios
    #########################
    # ew120esim = backtester(excess=(True,), rolling_window=(120,), programs=['equally_weighted'], save=False,
    #                        logreturns=False)  # 25 secs
    # ew120esim = backtester(excess=(False,), rolling_window=(120,), programs=['equally_weighted'], save=False,
    #                        logreturns=True)
    # cr = ew120esim[list(ew120esim.keys())[0]].cumulative_portfolio_returns()
    # plt.plot(cr)
    # plt.show()
    # ew120rsim = backtester(excess=(False,), rolling_window=(120,), programs=['equally_weighted'], save=False,
    #                       logreturns=False)  # 25 secs

    # ew120sim: dict = backtester(excess=(True, False), rolling_window=(120,), programs=['equally_weighted'],
    #                             save=False, logreturns=False)  # 1 min
    #
    # plt.figure(figsize=(9, 5))
    # plt.plot(ew120sim['120_excess_equally_weighted_sample'].cumulative_portfolio_returns(),
    #          color='k', label='Excess returns')
    # plt.plot(ew120sim['120_simple_equally_weighted_sample'].cumulative_portfolio_returns(),
    #          color='k', label='Returns', linestyle=':')
    # plt.legend()
    # plt.tight_layout()
    # plt.grid(True)
    # plt.savefig('C:/Users/YBant/Desktop/backtests_am_project/plots/120_excess_vs_normal_equally_weighted_sample.png',
    #             transparent=True, dpi=600)
    # plt.show()


    # Backtest for 120-period rolling windows over excess returns
    """
    Processing time:
    'mean_variance' & 'sample': 20 secs
                    & 'constant_correlation': 3.58 min
                    & 'single_index': 1.25 min
                    & 'identity': 1.20 min
    'equally_weighted' & 'sample': 30 secs
    'market_cap_weighted' & 'sample': 30 secs
    """
    # ptf120excess: dict = backtester(excess=(True, ), rolling_window=(120, ), programs=prgs)

    # Backtest for 30-period rolling windows over excess returns
    """
    Processing time:
    'mean_variance' & 'sample': 46 secs
                    & 'constant_correlation': 18 mins
                    & 'single_index': 6 mins
                    & 'identity': 6 mins
    'equally_weighted' & 'sample': 1 min
    'market_cap_weighted' & 'sample': 1 min 30 secs
    """
    # ptf30excess: dict = backtester(excess=(True,), rolling_window=(30,), programs=prgs)

    # Equally-Weighted Portfolio backtest
    # ew120b = backtester(excess=(True, False), rolling_window=(120, ), programs=['equally_weighted'])
    # plt.plot(ew120b['120_excess_equally_weighted_sample'].cumulative_portfolio_simple_returns())
    # plt.plot(ew120b['120_excess_equally_weighted_sample'].cumulative_portfolio_excess_returns())
    # plt.plot(ew120b['120_simple_equally_weighted_sample'].cumulative_portfolio_excess_returns())
    # plt.plot(ew120b['120_simple_equally_weighted_sample'].cumulative_portfolio_simple_returns())
    # plt.show()

    # Minimum-variance backtest for a 120-period rolling window
    #############################################################
    # minv120: dict = backtester(excess=(True, False), rolling_window=(120, ), programs=['min_variance'], save=True)
    """
        Processing time: 3 mins
        """
    # plt.plot(minv120['120_excess_min_variance_sample'].cumulative_portfolio_simple_returns(), label='excess cumsr')
    # plt.plot(minv120['120_simple_min_variance_sample'].cumulative_portfolio_simple_returns(), label='simple cumsr')
    # plt.legend()
    # plt.show()
    # same
    # plt.plot(minv120['120_excess_min_variance_sample'].cumulative_portfolio_excess_returns(), label='excess cumer')
    # plt.plot(minv120['120_simple_min_variance_sample'].cumulative_portfolio_excess_returns(), label='simple cumer')
    # plt.legend()
    # plt.show()

    # Verifying that excess and simple returns optimizations differ from each other
    #################################################################################
    # minv120sw = minv120['120_simple_min_variance_sample'].weights()  # [124x3303]
    # r = p.set_index('date').loc[minv120sw.index, minv120sw.columns].fillna(0)
    # r = r.pct_change().fillna(0).iloc[1:, :]  # [123x3303]
    # minv120w_shifted = minv120sw.shift(1).dropna()  # [123x3303]
    # ptf_r = (minv120w_shifted.mul(r)).sum(axis=1)  # [123x1]
    # cum_ptf_r = ((1 + ptf_r).cumprod()) - 1
    #
    # minv120ew = minv120['120_excess_min_variance_sample'].weights()  # [124x3303]
    # ptf_er = (minv120w_shifted.mul(r)).sum(axis=1) - np.ravel(rf.set_index('date').loc[r.index].to_numpy())
    # cum_ptf_er = ((1 + ptf_er).cumprod()) - 1

    # Export NAVs to feather
    ##########################
    # path = 'C:/Users/YBant/Desktop/backtests_am_project/navs/manual_minvar_120_ptf_cum_excess_simple.feather'
    # manual_minvar_120_ptf_cum_excess_simple = pd.concat([cum_ptf_r, cum_ptf_er], axis=1)
    # manual_minvar_120_ptf_cum_excess_simple.columns = ['manual_minvar_120_ptf_cum_simple',
    #                                                    'manual_minvar_120_ptf_cum_excess']
    # manual_minvar_120_ptf_cum_excess_simple.reset_index().head().rename({'index': 'date'}, axis=1).to_feather(path)

    # plt.figure(figsize=(15, 8))
    # plt.title("Minimum variance NAV curves for a 120-period rolling window")
    # plt.plot(cum_ptf_er, label='Excess returns', linewidth=2, color='k')
    # plt.plot(cum_ptf_r, label='Simple returns', linewidth=2, color='k', linestyle=':')
    # plt.plot(minv120['120_simple_min_variance_sample'].cumulative_portfolio_simple_returns(), label='simple cumsr')
    # plt.plot(minv120['120_excess_min_variance_sample'].cumulative_portfolio_excess_returns(), label='excess cumer')
    # plt.tight_layout()
    # plt.legend()
    # plt.grid(True)
    # # path = "C:/Users/YBant/Desktop/backtests_am_project/plots/"
    # # plt.savefig(f'{path}nav_minvar_120_excess_simple.png', transparent=True, dpi=600)
    # plt.show()

    # import porfolios from feather
    #################################
    # ew120 = pd.read_feather('../data/navs/navs_True_False_120_periods_equally_weighted.feather')
    # ew120.rename({'index': 'date'}, axis='columns', inplace=True)
    # ew120.set_index('date', inplace=True)
    # ew120.plot()
    # plt.show()

    # Backtesting the minimum variance portfolio with sample covariance estimation method
    #######################################################################################
    # min_var: dict = backtest_minvar(data, excess=True, rolling_window=30)
    # mvp_nav = min_var['min_variance_sample'].cumulative_portfolio_excess_returns()

    # Manual testing
    ##################
    # We want a (3303,) vector of optimal weights
    # minv120sw = minv120['120_simple_min_variance_sample'].weights()  # (124, 3303)
    # # We define a subset of prices such that no column (i.e. asset) has only NaN values
    # p_subset: np.ndarray = p.set_index('date').loc[minv120sw.index, minv120sw.columns].dropna(axis=1).to_numpy()  # (124, 1093)
    # rf_subset: np.ndarray = rf.set_index('date').loc[minv120sw.index].iloc[1:].to_numpy()  # (123, 1)
    # # Log-returns
    # lr: np.ndarray = np.diff(np.log(p_subset), axis=0)  # (123, 1093)
    # elr: np.ndarray = lr - rf_subset
    # assert not np.isinf(lr).any(), "'inf' values left."
    # mu: np.ndarray = np.apply_along_axis(np.nanmean, 0, lr)  # (1093,)
    # sigma: np.ndarray = np.cov(lr.T)  # (1093, 1093)
    # sigma_inv: np.ndarray = np.linalg.inv(sigma)
    # w_star = (1 / 3) * np.matmul(sigma_inv, mu)  # (1093,)
    #
    # # Simple returns
    # r = np.diff(p_subset) / p_subset[:, 1:]
    # assert not np.isinf(r).any(), "'inf' values left."
    #
    # p_subset2 = p_subset[:4, 2:7]  # (4, 5)
    # lr2 = np.diff(np.log(p_subset2), axis=0)  # (3, 5)
    # mu2 = np.apply_along_axis(np.nanmean, 0, lr2)  # (5,)
    # sigma2: np.ndarray = np.cov(lr2.T)  # (5, 5)
    # sigma2_inv: np.ndarray = np.linalg.inv(sigma2)  # (5, 5)

    # Manual testing for log returns EW
    #####################################

    # Test Markowitz
    ##################
    # w = pd.read_feather('../data/weights/120_simple_min_variance_sample.feather').set_index('date')  # (124, 3303)
    # p_subset: np.ndarray = p.set_index('date').loc[w.index, w.columns].dropna(axis=1).to_numpy()  # (124, 1093)
    # rf_subset: np.ndarray = rf.set_index('date').loc[w.index].iloc[1:].to_numpy()  # (123, 1)
    # w_star_markowitz_elog = Markowitz(p_subset, rf_subset).compute()  # (1093,)
    # w_star_markowitz_esim = Markowitz(p_subset, rf_subset, logreturns=False).compute()
    #
    # w_star_markowitz_log = Markowitz(p_subset).compute()
    # w_star_markowitz_sim = Markowitz(p_subset, logreturns=False).compute()

    # Test predict_weights_static
    ###############################
    # # Excess log returns | mean variance | sample covariance matrix
    # w_star_markowitz_elog_markow = OptimizedPortfolio.predict_weights_static(Markowitz, p_subset, rf_subset)
    # # Log returns | mean variance | sample covariance matrix
    # w_star_markowitz_log_markow = OptimizedPortfolio.predict_weights_static(Markowitz, p_subset)
    # # Simple returns | mean variance | Sample covariance matrix
    # w_star_markowitz_sim_markow = OptimizedPortfolio.predict_weights_static(Markowitz, p_subset, logreturns=False)
    # # Excess log returns | mean variance | constant correlation shrinkage covariance matrix
    # w_star_markowitz_elog_markowcc = OptimizedPortfolio.predict_weights_static(MarkowitzShrinkedConstantCorrelation,
    #                                                                            p_subset, rf_subset)
    # # Excess log returns | mean variance | shrinkage towards single index covariance matrix
    # w_star_markowitz_elog_markowsi = OptimizedPortfolio.predict_weights_static(MarkowitzShrinkedSI, p_subset, rf_subset)
    # # Excess log returns | mean variance | shrinkage towards identity covariance matrix
    # w_star_markowitz_elog_markowid = OptimizedPortfolio.predict_weights_static(MarkowitzShrinkedIdentity, p_subset,
    #                                                                            rf_subset)


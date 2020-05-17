import matplotlib.pyplot as plt
import pandas as pd
from main import backtester
from markowitz import OptimizedPortfolio
import time
import datetime
import os


def backtest_excess_vs_normal_navs(main_dir: str, dirpaths: dict, cmap: dict, color: str, save: bool = False,
                                   program: str = None, rolling_window: int = 120, linewidth: int = 1,
                                   visualize_backtest: bool = False) -> dict:

    kwargs: dict = dict(excess=(True, False), rolling_window=(rolling_window,), programs=[program],
                        logreturns=False, save=save, verbose=True, visualize_backtest=visualize_backtest,
                        dirpaths=dirpaths)

    result: dict = backtester(**kwargs)

    navs = pd.read_feather(f"{dirpaths['navs_dir']}{program}_{rolling_window}"
                           f"_excess_normal_simple_sample.feather").set_index('date')

    navs.columns = ['Excess', 'Normal']

    abbr: dict = {'mean_variance': 'MVO', 'equally_weighted': 'EW', 'market_cap_weighted': 'MC'}

    plt.style.use('classic')

    plt.figure(figsize=(9, 5), facecolor='w')
    plt.title(f"{abbr[program]} cumulative portfolio simple returns for a {rolling_window}-months rolling window")
    plt.plot(navs['Excess'].mul(100), label='Excess', linewidth=linewidth,
             color=cmap[color])
    plt.plot(navs['Normal'].mul(100), label='Normal', linewidth=linewidth,
             color=cmap[color], linestyle=':')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, ls='dashdot', alpha=0.5)
    plt.legend(loc='best', prop={'size': 12})
    plt.ylabel("%")
    plt.xlabel("Year")
    plt.tight_layout()
    if save:
        plt.savefig(f"{main_dir}plots/navs_{program}_excess_normal_{rolling_window}.png",
                    dpi=600, transparent=True)
    plt.show()

    return result


def backtest_minvar(data, excess: bool = True, rolling_window: int = 120, verbose: bool = True):
    portfolios_dict = dict()

    p, mv, rf = data.values()

    start = time.time() if verbose else None

    mvp = OptimizedPortfolio(prices=p.copy(), mv=mv.copy(), rf=rf.copy(),
                             excess=excess,
                             rolling_window=rolling_window,
                             scheme='min_variance',
                             cov_estimation_method='sample',
                             verbose=verbose)
    # mvp_nav = mvp.cumulative_portfolio_excess_returns()
    portfolios_dict["min_variance_sample"] = mvp

    if verbose:
        print(f"Processing time for min-variance 'sample': {(time.time() - start) / 60.:.2f} minutes.")

    return portfolios_dict


def backtest_all(main_dir: str, dirpaths: dict, cmap: dict, linewidth: int = 1,
                 excess: bool = True, rolling_window: int = 120, save: bool = False,
                 visualize_backtest: bool = False) -> dict:

    excess_str: str = "excess" if excess else "normal"

    programs: list = ['mean_variance', 'equally_weighted', 'market_cap_weighted']

    optptfs: dict = backtester(excess=(excess,), rolling_window=(rolling_window,), programs=programs,
                               logreturns=False, save=save, verbose=True, visualize_backtest=visualize_backtest,
                               dirpaths=dirpaths)

    navs = pd.read_feather(f"{dirpaths['navs_dir']}"
                           f'mean_variance_equally_weighted_market_cap_weighted_{rolling_window}'
                           f'_{excess_str}_simple_sample_constant_correlation_single_index_'
                           f'identity.feather').set_index('date')

    plt.style.use('classic')
    plt.figure(figsize=(9, 5), facecolor='w')
    plt.title(f"{excess_str.capitalize()} cumulative portfolio simple returns for a {rolling_window}-months "
              f"rolling window")
    plt.plot(navs[f'mean_variance_{rolling_window}_portfolio_{excess_str}_simple_returns_sample'].mul(100),
             label='Mean Variance (Sample)', linewidth=linewidth, color=cmap['mvo_sample'])
    plt.plot(navs[f'mean_variance_{rolling_window}_portfolio_{excess_str}_simple_returns_constant_correlation'].mul(100),
             label='Mean Variance (Constant Correlation)', linewidth=linewidth, color=cmap['mvo_cc'])
    plt.plot(navs[f'mean_variance_{rolling_window}_portfolio_{excess_str}_simple_returns_single_index'].mul(100),
             label='Mean Variance (Single-Index)', linewidth=linewidth, color=cmap['mvo_si'])
    plt.plot(navs[f'mean_variance_{rolling_window}_portfolio_{excess_str}_simple_returns_identity'].mul(100),
             label='Mean Variance (Identity)', linewidth=linewidth, color=cmap['mvo_id'])
    plt.plot(navs[f'equally_weighted_{rolling_window}_portfolio_{excess_str}_simple_returns_sample'].mul(100),
             label='Equally-Weighted', linewidth=linewidth, color=cmap['ew'])
    plt.plot(navs[f'market_cap_weighted_{rolling_window}_portfolio_{excess_str}_simple_returns_sample'].mul(100),
             label='Market-Cap Weighted', linewidth=linewidth, color=cmap['mc'])
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, ls='dashdot', alpha=0.5)
    plt.legend(loc='best', prop={'size': 12})
    plt.ylabel("%")
    plt.xlabel("Year")
    plt.tight_layout()
    if save:
        plt.savefig(f"{main_dir}plots/navs_{excess_str}_{rolling_window}.png", dpi=600, transparent=True)
    plt.show()

    return optptfs


def plot_rf_vs_ff(main_dir: str):
    # monthly GS1M vs. F-F RF (their 1-month TBill return is from Ibbotson and Associates)
    rf = pd.read_feather('../data/clean/rf.feather')
    rf.set_index('date', inplace=True)
    rf = rf.mul(100)
    rf_ff = pd.read_csv('../data/test/F-F_Research_Data_Factors.CSV', skiprows=3, nrows=1124)
    rf_ff.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    rf_ff.drop(rf_ff.columns.difference(['date', 'RF']), 1, inplace=True)
    rf_ff.date = rf_ff.date.astype(str)
    rf_ff.date = rf_ff.date.apply(lambda x: datetime.datetime.strptime(x, '%Y%m'))
    rf_ff.set_index('date', inplace=True)
    rf_ff = rf_ff

    # Get the shortest sample initial date
    start = max([rf.index[0], rf_ff.index[0]])
    plt.style.use('classic')

    plt.figure(figsize=(9, 5), facecolor='w')
    plt.plot(rf_ff.loc[start:], label='Fama-French')
    plt.plot(rf.loc[start:], label='GS1M')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, ls='dashdot', alpha=0.5)
    plt.legend(loc='best', prop={'size': 6})
    plt.ylabel("%")
    plt.xlabel("Year")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{main_dir}plots/monthly_rf_vs_rf_fama_french.png', dpi=600, transparent=True)
    plt.show()


def import_portfolios():
    """
    Import portfolios that come from optimization.
    """
    current_dir = os.path.dirname(__file__)
    names = ['cum_porfolio_excess_returns', 'cum_porfolio_simple_returns',
             'porfolio_excess_returns', 'porfolio_simple_returns', 'ew_weights', 'mc_weights',
             'meanvar_constant_correlation_weights', 'meanvar_identity_weights',
             'meanvar_sample_weights', 'meanvar_single_index_weights', 'minvar_sample_weights']

    filepaths = [f'../data/portfolios/{name}.feather' for name in names]
    filenames = {k: os.path.join(current_dir, v) for (k, v) in zip(names, filepaths)}
    files = {k: pd.read_feather(v) for (k, v) in filenames.items()}
    files = {k: v.set_index('index').rename_axis('date') for k, v in files.items()}
    return files
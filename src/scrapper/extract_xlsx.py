from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import datetime
from pandas_datareader.data import DataReader
import numpy as np
import re
from functools import reduce
from utils import assert_right_type
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

pd.set_option('mode.chained_assignment', 'warn')  # 'raise', None


class PriceData(ABC):
    @staticmethod
    def load(filename: str, rf=True) -> PriceData:
        xlsx: pd.ExcelFile = pd.ExcelFile(filename)
        prices: pd.DataFrame = pd.read_excel(xlsx, "Prices")
        mv: pd.DataFrame = pd.read_excel(xlsx, "MV")
        return PriceData(prices, mv, rf)

    def __init__(self, prices: pd.DataFrame, mv: pd.DataFrame, rf: bool):
        price_names = np.array([price_name for price_name in prices["NAME"]])
        mv_names = np.array([mv_name.split(" - ")[0] for mv_name in mv["NAME"]])
        assert all([price_name == mv_name for price_name, mv_name in zip(price_names, mv_names)])
        self.__company_names: np.array = price_names
        del mv["NAME"]
        del prices["NAME"]
        self.__prices: pd.DataFrame = prices.interpolate()
        self.__mv: pd.DataFrame = mv.interpolate()
        self.__moving_portfolios: pd.DataFrame = self.__compute_portfolios()
        self.__dates: np.ndarray = self.__moving_portfolios.columns.values
        self.__rf = self.import_rf().interpolate() if rf else None

    def __compute_portfolios(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()
        for date in self.__mv:
            total = np.sum(self.__mv[date])
            df[date] = np.array([num / total for num in self.__mv[date]])
            end_sum = np.sum(df[date])
            assert (100 - 1e5 < end_sum < 100 + 1e5)
        return df

    def moving_portfolios(self) -> pd.DataFrame:
        return self.__moving_portfolios

    def prices(self) -> pd.DataFrame:
        return self.__prices

    def names(self) -> np.array:
        return self.__company_names

    def name(self, index: int) -> str:
        return self.__company_names[index]

    def set_rf(self, ticker: str, data_source: str):
        self.__rf = self.import_rf(ticker, data_source)

    def import_rf(self, ticker='DGS1MO', data_source='fred'):
        start = self.__dates.copy()[0]
        end = self.__dates.copy()[-1]
        df: pd.DataFrame = DataReader(ticker, data_source, start=start, end=end).resample('M').last().div(100)
        df.interpolate(inplace=True)
        df.iloc[0] = np.nan  # because returns have one observation less than prices
        return df

    def get_rf(self) -> pd.DataFrame:
        return deepcopy(self.__rf)


class DataAnalyser:

    __pattern = re.compile('^Name*')

    def __init__(self, dataset: pd.DataFrame):
        assert isinstance(dataset, pd.DataFrame), "The dataset passed as argument must be of 'pandas.DataFrame' type"
        assert self.__class__.__pattern.match(dataset.columns[0]), "The first columns does not begin with 'Name'"
        self.__data = dataset

    def info(self):
        return self.__data.info()

    def get_dataset(self):
        return deepcopy(self.__data)

    def describe(self):
        return self.__data.describe()

    def display_columns_of_type(self, dtype_=object) -> pd.DataFrame:
        """Display a DataFrame with columns of a certain type."""
        return self.__data[[col for col in self.__data.columns if self.__data[col].dtype == dtype_]]

    def error_counter(self, start_yr: int = 2000) -> pd.DataFrame:
        # Empty DataFrame
        error_counts_df = pd.DataFrame([], columns=['Year', '#ERROR count'])

        # Extraction of the labels of the yearly index constituents
        names = self.__data.columns[self.__data.columns.str.count(r'(^Name*)') == 1]

        for i, name in enumerate(names):
            error_counts = self.count_errors(self.__data, column_label=name)
            yr = i + start_yr
            error_counts_df = error_counts_df.append({'#ERROR count': error_counts, 'Year': yr}, ignore_index=True)

        return error_counts_df

    def list_of_frames(self) -> list:
        """Returns a list of DataFrames.

        Knowing the pattern of our DataFrame is Name, date1 to date 12, and so on, we can use a for loop
        incrementing 13 as a constant.
        """
        return [self.__data.iloc[:, i:i + 13].copy() for i in range(0, len(self.__data.columns), 13)]

    def display_nan(self):
        """Display NaNs for each column
        """
        return self.__data.isna().sum()

    def remove_errors(self) -> list:
        return [df[~df.filter(regex='(^Name*)', axis=1).isin(["#ERROR"]).iloc[:, 0]] for df in self.list_of_frames()]

    def tidy_list(self) -> list:
        """Clean list of DataFrames
        """
        # list of DataFrames without error messages
        frames_li = self.remove_errors()

        # list with DataFrames being of the right type
        corrected_types_frames_li = [self.convert_types(df) for df in frames_li]

        # Check whether the values in the row are valid
        error_message = "There are either null or invalid data left in a DataFrame from the list."
        for frame in corrected_types_frames_li:
            assert frame.iloc[:, 1:].apply(self.check_null_or_valid, axis=1).all().all(), error_message

        return corrected_types_frames_li

    def name_spellings_for_list_of_df(self, regex: str = " DEAD \-*", mask_inverse: bool = True):
        """Check invalid company names in the list of DataFrames that have already been cleaned from
        errors and wrong types. Returns a list of Series containing the invalid names.
        """
        # List of tidy DataFrames
        li = self.tidy_list()

        # Column labels
        col_labels = ['Name%s%s' % ('.', str(i)) if (i > 0) else 'Name' for i in range(len(li))]

        return [self.name_spellings(df,
                                    regex=regex,
                                    col_label=col_labels[i],
                                    mask_inverse=mask_inverse) for i, df in enumerate(li)]

    @staticmethod
    def name_spellings(df: pd.DataFrame, regex: str = " DEAD \-*", col_label: str = 'Name',
                       mask_inverse: bool = True):
        """Lists the invalid names according to a regular expression for a DataFrame passed as argument"""
        # Create the series of countries: countries
        stocks = df[col_label]

        # Drop all the duplicates from countries
        stocks = stocks.drop_duplicates().dropna()

        pattern = re.compile(regex)

        # Create the Boolean vector: mask
        mask = stocks.str.contains(pattern)

        # Invert the mask: mask_inverse
        if mask_inverse:
            mask = ~mask

        # Subset countries using mask_inverse: invalid_countries
        invalid_companies = stocks[mask]

        return invalid_companies

    @staticmethod
    def check_null_or_valid(row_data):
        """Function that takes a row of data,
        drops all missing values,
        and checks if all remaining values are greater than or equal to 0
        """
        no_na = row_data.dropna()
        numeric = pd.to_numeric(no_na)
        ge0 = numeric >= 0
        return ge0

    @staticmethod
    def count_errors(df: pd.DataFrame, column_label: str):
        counts = df[str(column_label)].value_counts()
        return counts.loc['#ERROR'] if '#ERROR' in counts.index else 0

    @staticmethod
    def convert_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert all columns that are not 'Name' to float only if the types are 'int' and 'float'
        """
        # Select the columns to inspect, i.e., the ones who have price time series and that are not of type float
        # cols_to_inspect: pd.Series = df.dtypes[(df.dtypes != float) & df.dtypes.index.map(lambda s: type(s) != str)]
        # Alternatively:
        cols_to_inspect: pd.Series = df.dtypes[(df.dtypes != float)].filter(regex='^(?!Name).*')

        # Select the labels of the columns that are not of type 'float'
        cols_li: list = cols_to_inspect.index.to_list()

        # Count the type occurrences for each column (list of pd.Series)
        type_counts_per_col: list = [df[col].apply(lambda x: type(x).__name__).value_counts() for col in cols_li]

        df = df.copy()  # to avoid 'SettingWithCopyWarning' ambiguity
        for i, col in enumerate(cols_li):
            # if the types found in the column subject to inspection are 'int' or 'float' then convert to all to 'float'
            if type_counts_per_col[i].index.isin(['float', 'int']).all():
                df[col] = df[col].astype('float')
        return df


class DataCleaner(DataAnalyser):
    def __init__(self, dataset: pd.DataFrame, regex: str = ' DEAD -'):
        super().__init__(dataset)
        self.__regex = regex

    def cleaned_list(self, regex: str = None):
        if not regex:
            regex = self.__regex

        # List of DataFrames without error messages and with correct types
        tl1 = self.tidy_list()

        # Rename all columns beginning with 'Name...' to 'Name'
        tl2 = [df.rename(columns={df.columns[0]: "Name"}) for df in tl1]

        # Create 2 columns 'company' and 'dead' and add them to each DataFrame in 'tl2', plus drop 'Name' column
        tl3 = list()
        for frame in tl2:
            df = frame.copy()
            df['str_split'] = df.Name.str.split(regex)
            # Create new column with clean company names
            df['company'] = df.str_split.str.get(0)
            # Create column with dead events and their information
            df['dead'] = df.str_split.str.get(1)
            # Drop the dirty 'Name' column
            df.drop(["Name", "str_split"], inplace=True, axis=1)

            # Get rif of 'MARKET VALUE' residual string in company names
            df['str_split'] = df.company.str.split(" - ")
            df['company'] = df.str_split.str.get(0)
            df.drop(["str_split"], inplace=True, axis=1)

            # Get rid of ' SUSP' string in company names
            df['str_split'] = df.company.str.split(" SUSP")
            df['company'] = df.str_split.str.get(0)
            df.drop(["str_split"], inplace=True, axis=1)

            # Drop duplicate observations among the company names
            df = df.drop_duplicates(subset=["company"])
            # Drop missing values among the companies
            df.dropna(subset=["company"], inplace=True)
            # Set company to index to sort them easier
            df.set_index("company", inplace=True)
            # Ascending sort
            df.sort_values("company", inplace=True)
            # Reset index
            df.reset_index(inplace=True)
            tl3.append(df)

        return tl3

    def melted_list(self, regex: str = None) -> list:
        """Melt the cleaned DataFrames. Returns a list with melted DataFrames"""
        if not regex:
            regex = self.__regex

        # Compile the list of DataFrames to melt
        li = self.cleaned_list(regex=regex)

        return [pd.melt(frame=df.drop('dead', axis='columns'),
                        id_vars='company',
                        var_name='date',
                        value_name='price') for df in li]

    def save_as_feather(self, path: str = '../../data/clean/msci_world.feather', regex: str = None):
        if not regex:
            regex = self.__regex
        clean_data = self.grand_dataset(regex=regex)
        return clean_data.reset_index().to_feather(path)

    def grand_dataset(self, regex: str = None):
        if not regex:
            regex = self.__regex
        # Stack vertically all the clean DataFrames and resets the index
        concatenated: pd.DataFrame = pd.concat(self.melted_list(regex=regex), ignore_index=True)
        # Return a DataFrame displaying price series by companies
        return concatenated.pivot_table(values='price', index='date', columns='company')

    def create_window(self, start: str = '2000', end: str = '2005') -> pd.DataFrame:
        # begin with the grand matrix with outer join and then select by month
        grand: pd.DataFrame = self.grand_dataset()
        # selects a time frame of which only companies that have lived all this time are kept
        return grand.loc[start:end].dropna(axis=1)


class DataLoader:

    def load_feather(self, filename: str, kind: str):
        if kind == "P":  # price series
            prices: pd.DataFrame = pd.read_feather(filename)
            if "date" in prices.columns:
                prices.set_index("date", inplace=True)
            self.__prices = prices
            self.__update()
        elif kind == "MV":  # market value series
            mv: pd.DataFrame = pd.read_feather(filename)
            if "date" in mv.columns:
                mv.set_index("date", inplace=True)
            self.__mv = mv
            self.__update()
        elif kind == "RF":  # risk-free rate series
            self.__rf = pd.read_feather(filename)
            self.__update()
        else:
            print("'load_feather' method has not loaded any dataset.")

    def __init__(self, prices: pd.DataFrame = None, mv: pd.DataFrame = None, rf: pd.Series or pd.DataFrame = None):
        if prices is not None:
            self.__set_dateindex(prices)

        if mv is not None:
            self.__set_dateindex(mv)

        if rf is not None:
            self.__set_dateindex(rf)

        if (prices is not None) and (mv is not None):
            # Make columns match
            self.__col_match(prices, mv)
            self.__company_names: np.ndarray = prices.columns.to_numpy()

        # affectations
        self.__prices: pd.DataFrame = prices
        self.__mv: pd.DataFrame = mv
        self.__rf: pd.Series = rf
        self.__date = None
        self.__update()
        self.__dataset_dict_getter = {'p': self.get_prices(), 'mv': self.get_mv(), 'rf': self.get_rf()}
        self.__dataset_dict_setter = {'p': self.__prices, 'mv': self.__mv, 'rf': self.__rf}
        self.__t, self.__n = self.__prices.shape

    def __set_dateindex(self, df: pd.DataFrame or pd.Series):
        # Verify if the type of the argument is right
        err_msg = f'\'pd.DataFrame\' or \'pd.Series\' expected type, not \'{type(df).__name__}\'.'
        assert (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)), err_msg

        if isinstance(df, pd.DataFrame):
            if df.reset_index().select_dtypes(np.datetime64).shape[1] == 1:
                # if the column with dates is the index, rename the index as 'date'
                if isinstance(df.index, pd.DatetimeIndex):
                    df.index.name = 'date'
                # if the column with dates is not the index, set it as it and rename it
                else:
                    # Set column as index
                    column_label = df.select_dtypes(np.datetime64).columns[0]
                    df.set_index(column_label, inplace=True)
                    df.index.name = 'date'
            elif df.reset_index().select_dtypes(np.datetime64).shape[1] > 1:
                raise Exception("There are too much datetime dtype columns in the dataset.")
            else:
                raise Exception("There is no datetime dtype column in the dataset.")
        elif isinstance(df, pd.Series):
            df = df.to_frame()
            self.__set_dateindex(df)

    def get_t(self) -> int:
        """:return Returns the number of observations of the whole dataset."""
        return self.__t

    def get_n(self) -> int:
        """:return Returns the number of stocks of the whole dataset."""
        return self.__n

    def create_window(self, start=None, end=None, key='p') -> pd.DataFrame:
        # If start/end are None, then assign to them the first/last dates of the dataset
        start, end = self._check_start_end(start, end)

        # First, subset the data by the timeframe
        # Second, drop all columns (i.e. companies) that exhibit missing values (i.e. nan)
        subset: pd.DataFrame = self.__dataset_dict_getter[key].loc[start:end].dropna(axis=1)
        return subset

    def get_rf(self):
        return deepcopy(self.__rf)

    def get_prices(self):
        return deepcopy(self.__prices)

    def get_mv(self):
        return deepcopy(self.__mv)

    def get_company_names(self):
        return deepcopy(self.__company_names)

    def get_dates(self):
        return deepcopy(self.__date)

    def __update(self):
        """Updates class attributes."""
        p, mv, rf = self.__prices, self.__mv, self.__rf
        # Select attributes different from 'None'
        li = [x for x in (p, mv, rf) if x is not None]

        # if there is no element in the list, i.e., if all attributes are 'None'
        if len(li) == 0:
            self.__date = None
        # if there is only one element not 'None' in the list, 'self.__date' should be equal to its index
        elif len(li) == 1:
            self.__date: np.ndarray = li[0].index.to_numpy()
        # if there is at least 2 attributes that are not 'None' we must verify if rows match in length and in values
        else:
            # if lengths match (to prevent ValueError)
            if self.__check_index_length_match(li):
                # if length and values are the same
                if self.__check_index_values_match(li):
                    self.__date = li[0].index.to_numpy().copy()
                # if lengths are equal among each dataset index, but not the values
                else:
                    # if values do not match, we force them to take the same
                    print("Lengths of rows match, but not they have different values.")
                    self.__date = li[0].index.to_numpy().copy()
                    self.__make_indices_values_match()
                    assert self.__check_index_values_match(li)
            # if any length mismatch, we truncate all DataFrames or Series
            else:
                # Get the oldest date among the list of DataFrames
                min_date = min([df.index.min() for df in li])
                # In the case there is a risk-free rate and that it begins after the other series: try
                # to complete it with the 3 month proxy
                if (self.__rf is not None) & (self.__rf.index[0] > min_date):
                    # Get initial date of the risk-free rate series
                    end = rf.index[0]
                    # 3-Month Treasury Constant Maturity Rate (GS3M)
                    rf3m = DataReader('GS3M', 'fred', start=min_date, end=end).resample('MS').mean()
                    # We have to drop the last row to prevent overlapping
                    # We couldn't have used timedelta to go back 1 month as some have 31 days while others 30
                    rf3m.drop(rf3m.tail(1).index, inplace=True)
                    rf3m.columns = rf.columns
                    rf3m = rf3m.div(100).div(12)
                    # Concatenate both risk-free rates pd.Series
                    rf_concat = pd.concat([rf3m, self.__rf], sort=True)
                    errmsg: str = f"Got {rf_concat.shape} shape, but ({len(li[0].index)}, 1) expected."
                    assert rf_concat.shape[1] == 1, errmsg
                    self.__rf = rf_concat
                    # Join both series in a sole one
                    # self.__rf = rf_concat.iloc[:, 0].add(rf_concat.iloc[:, 1], fill_value=0)
                else:
                    # Truncate rows of different length according to their dates
                    self.__truncate_rows()
                    # Verify if the rows were correctly truncated
                    not_none_attributes_list = self.__among_not_none_attributes()
                    err_message = "Rows were not correctly truncated"
                    assert self.__check_index_length_match(not_none_attributes_list), err_message
                    # Update the 'self.__date' attribute with the first item
                    self.__date = not_none_attributes_list[0].index.to_numpy().copy()
                    # Propagate same indexes to the other datasets to force a perfect match
                    self.__make_indices_values_match()

                    # Verify that indices have same indexes
                    err_message = "Values do not match among not 'None' attributes."
                    assert self.__check_index_values_match(self.__among_not_none_attributes()), err_message
                self.__update()

    def __among_not_none_attributes(self) -> list:
        return [x for x in (self.__prices, self.__mv, self.__rf) if x is not None]

    def __truncate_rows(self):
        """Truncate rows according to the most recent starting dates among datasets and the oldest
        ending dates among datasets."""
        p, mv, rf = self.get_prices(), self.get_mv(), self.get_rf()
        li = [p, mv, rf]
        start_date, end_date = self.__get_start_end_dates_from_datetimeindex(li)
        self.__prices = p.truncate(before=start_date, after=end_date, copy=False) if p is not None else None
        self.__mv = mv.truncate(before=start_date, after=end_date, copy=False) if mv is not None else None
        self.__rf = rf.truncate(before=start_date, after=end_date, copy=False) if rf is not None else None

    def __make_indices_values_match(self):
        self.__prices.index = self.__date.copy() if self.__prices is not None else None
        self.__mv.index = self.__date.copy() if self.__prices is not None else None
        self.__rf.index = self.__date.copy() if self.__prices is not None else None

    @staticmethod
    def __check_index_length_match(li: list) -> bool:
        """Verifies if lengths of all indices of DataFrames contained in list are equal or not."""
        return all([len(e.index) == len(li[0].index) for e in li])

    @staticmethod
    def __check_index_values_match(li: list) -> bool:
        return all([e.index.isin(li[0].index).all() for e in li])

    @staticmethod
    def __get_start_end_dates_from_datetimeindex(li: list) -> tuple:
        """Returns the most recent old date and the oldest recent date that all DataFrames in the list have in
        common."""
        start_date = max([x.index.min() for x in li])
        end_date = min([x.index.max() for x in li])
        return start_date, end_date

    def __col_match(self, df1, df2):
        df2_names_not_in_df1, df1_names_not_in_df2, match1 = self.__check_column_matching(df1, df2)
        if not match1:
            df1.drop(df1_names_not_in_df2, inplace=True, axis=1)  # modifies attribute!
            df2.drop(df2_names_not_in_df1, inplace=True, axis=1)  # modifies attribute!
        _, _, match2 = self.__check_column_matching(df1, df2)
        assert match2, "It seems that columns of both DataFrames do not match while they should even after column drop."

    @staticmethod
    def __check_column_matching(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
        """Checks if both DataFrames have the same columns

        Args:
            df1 (pd.DataFrame): DataFrame no 1
            df2 (pd.DataFrame): DataFrame no 2

        Returns:
            __check_column_matching: A tuple with names that are non common to both DataFrames and a boolean
        """
        col_names1 = np.array([name for name in df1.columns])
        col_names2 = np.array([name for name in df2.columns])

        names1 = [name for name in col_names2 if name not in col_names1]
        names2 = [name for name in col_names1 if name not in col_names2]

        return names1, names2, (names1 and names2) == []

    def _check_start_end(self, start, end):
        start = self.get_dates()[0] if start is None else start
        end = self.get_dates()[-1] if end is None else end
        return start, end


class Portfolio(DataLoader):
    def __init__(self, prices: pd.DataFrame = None, mv: pd.DataFrame = None, rf: pd.Series or pd.DataFrame = None,
                 logreturns: bool = True):
        super().__init__(prices, mv, rf)

        assert_right_type(logreturns, bool)
        self.__is_log = logreturns
        self.__return_version = {True: Portfolio.log_returns, False: Portfolio.simple_returns}

    @staticmethod
    def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change().dropna(axis=0, how='all')

    @staticmethod
    def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return np.log1p(prices).diff().dropna(axis=0, how='all')

    # def simple_returns(self, start=None, end=None) -> pd.DataFrame:
    #     """Simple returns DataFrame."""
    #     # If start/end are None assign to them the first/last dates of the dataset
    #     start, end = self._check_start_end(start, end)
    #
    #     prices_window: pd.DataFrame = self.create_window(start, end, key='p')
    #     return self.__simple_returns(prices_window)

    def returns(self, start=None, end=None, adaptive: bool = True) -> pd.DataFrame:
        """Compute adaptive returns according to log or simple returns initialization."""
        # If start/end are None assign to them the first/last dates of the dataset
        start, end = self._check_start_end(start, end)

        # Creates the adaptive window of prices
        # prices_window: pd.DataFrame = self.create_window(start, end, key='p')
        prices_window: pd.DataFrame = self.create_window(start, end, key='p') if adaptive else self.get_prices().loc[start:end, :]

        # Computes the price returns as specified (log or simple returns)
        return self.__return_version[self.__is_log](prices_window)

    def excess_returns(self, start=None, end=None, adaptive: bool = True) -> pd.DataFrame:
        """Adaptive returns matrix in excess of the risk-free rate."""
        # If start/end are None assign to them the first/last dates of the dataset
        start, end = self._check_start_end(start, end)

        # Simple or log-returns
        returns_window: pd.DataFrame = self.returns(start, end, adaptive=adaptive)

        rf_window: pd.DataFrame = self.create_window(returns_window.index[0], end, key='rf')
        return returns_window.sub(rf_window.to_numpy(), axis='columns')

    @staticmethod
    def _cumulative_returns(returns: pd.DataFrame, is_log: bool):
        return returns.cumsum() if is_log else returns.add(1).cumprod().sub(1)

    def cumulative_rf(self, start: str = '2000-1-1', end: str = '2005-12-1'):
        rf: pd.DataFrame = self.create_window(start, end, key='rf')
        cum_rf: pd.DataFrame = self._cumulative_returns(rf)
        cum_rf.columns = ['cumulative_rf']
        return cum_rf

    def get_islog(self) -> bool:
        return self.__is_log

    @staticmethod
    def plotter(*args, title="Portfolio NAV"):
        """Takes a list of pd.DataFrames and plots them."""
        reduce(lambda x, y: pd.merge(left=x, right=y, on='date'), args).plot(title=title)
        plt.grid(True)
        plt.show()


def data_loader_builder(verbose: bool = True):
    msci_prices_from_feather = pd.read_feather('../../data/clean/msci_world_prices.feather')
    if verbose:
        print("MSCI World Prices:")
        msci_prices_from_feather.info()
    msci_mv_from_feather = pd.read_feather('../../data/clean/msci_world_mv.feather')
    if verbose:
        print("\nMSCI World Market Values:")
        msci_mv_from_feather.info()

    # 1-Month Treasury Constant Maturity Rate (GS1M)
    rf_ = DataReader('GS1M', 'fred', start=datetime.datetime(1990, 1, 1)).resample('MS').mean()
    # We bring the annual rate to a monthly one
    rf_m = rf_.div(100).div(12)
    dl = DataLoader(prices=msci_prices_from_feather, mv=msci_mv_from_feather, rf=rf_m)
    prices, mv, rf = dl.get_prices(), dl.get_mv(), dl.get_rf()
    if verbose:
        print(f'\n\'prices\' shape: {prices.shape}')
        print(f'\'mv\' shape: {mv.shape}')
        print(f'\'rf\' shape: {rf.shape}')

    return rf, prices, mv, dl


def export_rf():
    rf, p, m, dl = data_loader_builder()
    rf = rf.reset_index()
    rf.columns.values[0] = 'date'
    rf.to_feather('../../data/clean/rf.feather')
    plt.plot(rf.date, rf.GS1M)
    plt.show()


if __name__ == '__main__':
    pass

    # export_rf()


    # # Annualized rf
    # rf = DataReader('GS1M', 'fred', start=datetime.datetime(1990, 1, 1))
    # rf_m = rf.div(100).div(12)
    # rf_m2 = rf.div(100).apply(lambda r: (1 + r/12) ** (1/12) - 1)
    # rf_m3 = rf.div(100).apply(lambda r: (1 + r) ** (1 / 12) - 1)
    # rf_ff = pd.read_csv('../../data/test/F-F_Research_Data_Factors.CSV', skiprows=3, nrows=1124)
    # rf_ff.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    # rf_ff.drop(rf_ff.columns.difference(['date', 'RF']), 1, inplace=True)
    # rf_ff.date = rf_ff.date.astype(str)
    # rf_ff.date = rf_ff.date.apply(lambda x: datetime.datetime.strptime(x, '%Y%m'))
    # rf_ff.set_index('date', inplace=True)
    # rf_ff = rf_ff.div(100)
    #
    #
    # start = max([rf1.index[0], rf_m.index[0]])
    # plt.plot(rf1.loc[start:], label='rf1')
    # plt.plot(rf_m.loc[start:], label='rf_m')
    # plt.plot(rf_m2.loc[start:], label='rf_m2')
    # plt.plot(rf_ff.loc[start:], label='rf_ff')
    # plt.plot(rf_m3.loc[start:], label='rf_m3')
    # plt.legend()
    # plt.show()

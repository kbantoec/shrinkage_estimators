import unittest
import pandas as pd
import os
from markowitz import Markowitz, MarkowitzShrinkedIdentity, MarkowitzShrinkedSI, MarkowitzShrinkedConstantCorrelation
from scrapper.extract_xlsx import DataLoader


class MarkowitzTest(unittest.TestCase):
    def setUp(self) -> None:
        current_dir = os.path.dirname(__file__)
        filename_prices = os.path.join(current_dir, '../data/clean/msci_world_prices.feather')
        filename_mv = os.path.join(current_dir, '../data/clean/msci_world_mv.feather')
        filename_rf = os.path.join(current_dir, '../data/clean/rf.feather')

        # import data
        p = pd.read_feather(filename_prices)
        mv = pd.read_feather(filename_mv)
        rf = pd.read_feather(filename_rf)

        data = DataLoader(prices=p, mv=mv, rf=rf)

        prices_pd = data.create_window('2003', '2016', 'p')
        self.prices = prices_pd.to_numpy()

    def test_sample(self):
        markowitz = Markowitz(self.prices)
        w = markowitz.compute()
        self.assertEqual(w.shape[0], self.prices.shape[1])

    def test_constant_correlation(self):
        markowitz = MarkowitzShrinkedConstantCorrelation(self.prices)
        w = markowitz.compute()
        self.assertEqual(w.shape[0], self.prices.shape[1])

    def test_identity(self):
        markowitz = MarkowitzShrinkedIdentity(self.prices)
        w = markowitz.compute()
        self.assertEqual(w.shape[0], self.prices.shape[1])

    def test_si(self):
        markowitz = MarkowitzShrinkedSI(self.prices)
        w = markowitz.compute()
        self.assertEqual(w.shape[0], self.prices.shape[1])


if __name__ == "__main__":
    test = MarkowitzTest()
    test.setUp()
    test.test_constant_correlation()
    test.test_si()


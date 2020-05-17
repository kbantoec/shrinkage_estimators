from flask import Flask, request
from markowitz import Markowitz, MarkowitzShrinkedConstantCorrelation, MarkowitzShrinkedIdentity, MarkowitzShrinkedSI
import json
import datetime
import os
import pandas as pd
import numpy as np
from scrapper.extract_xlsx import DataLoader
from pandas_datareader.data import DataReader
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def load_data():
    current_dir = os.path.dirname(__file__)
    filename_prices = os.path.join(current_dir, '../data/clean/msci_world_prices.feather')
    filename_mv = os.path.join(current_dir, '../data/clean/msci_world_mv.feather')

    p = pd.read_feather(filename_prices)
    mv = pd.read_feather(filename_mv)
    rf = DataReader('DGS1MO', 'fred', start=datetime.datetime(1990, 1, 1)).resample('MS').mean().div(100)

    return DataLoader(prices=p, mv=mv, rf=rf)


DATA = load_data()  # Pas le plus beau, mais ca ira pour les 4 routes qu'on doit faire


@app.route('/company_names')
def company_names():
    from_year = request.args.get('from') or '2003'
    to_year = request.args.get('to') or '2005'
    return json.dumps(list(DATA.get_company_names(from_year, to_year)))


estimators = {
    "markowitz": Markowitz,
    "constant_correlation": MarkowitzShrinkedConstantCorrelation,
    "identity": MarkowitzShrinkedIdentity,
    "single_index": MarkowitzShrinkedSI
}


@app.route("/<method>")
def compute_method(method: str):
    from_year = request.args.get('from') or '2003'
    to_year = request.args.get('to') or '2005'
    companies_query = request.args.get('companies')
    companies = companies_query.split(",") if companies_query is not None else None
    prices_pd = DATA.create_window(from_year, to_year, 'p', companies)
    prices = prices_pd.to_numpy()
    method = estimators[method]
    markowitz = method(prices)
    w = markowitz.compute()
    w_returns = np.apply_along_axis(np.sum, 0, (w * markowitz.log_returns()))
    return json.dumps(list(np.cumsum(w_returns)))

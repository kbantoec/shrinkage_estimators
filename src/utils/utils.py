import os
import pandas as pd


def pair(x: float) -> bool:
    return x % 2 == 0


def is_multiple_of(x: float, multiple: float) -> bool:
    return x % multiple == 0


def assert_right_type(o, c):
    """Verify if the object 'o' is an instance of the class 'c'
    and if not, raise an AssertionError with a customized message."""
    class_name: str = c.__name__ if isinstance(c, type) else type(c).__name__
    if not isinstance(o, c):
        assert isinstance(o, c), f"'{class_name}' object expected, but got '{type(o).__name__}' instead."


def err_msg(arg_el: str, arg_name: str, args_li: list):
    return f"Invalid '{arg_el}' {arg_name} " \
                  f"method. Pick one of the following instead: " \
                  + ', '.join(map(lambda x: "%s%s%s" % ("'", x, "'"), args_li)) + "."


def import_data():
    current_dir = os.path.dirname(__file__)
    filename_prices = os.path.join(current_dir, '../../data/clean/msci_world_prices.feather')
    filename_mv = os.path.join(current_dir, '../../data/clean/msci_world_mv.feather')
    filename_rf = os.path.join(current_dir, '../../data/clean/rf.feather')

    # import data
    p = pd.read_feather(filename_prices)
    mv = pd.read_feather(filename_mv)
    rf = pd.read_feather(filename_rf)
    return p, mv, rf
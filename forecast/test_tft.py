from datetime import datetime, timedelta
from pytz import timezone
import pandas as pd

from tft import set_time_idx

cache = pd.read_csv('data/ReconciledInjectionAndOfftake_201604_20170624_100756.csv')

def test_set_time_idx():
    df = cache.copy()
    df = df.where(df["PointOfConnection"] == 'OHB2201').sort_values(["TradingDate", "TradingPeriod"]).dropna()

    df = set_time_idx(df)

    assert df["time_idx"].min() == 0

# def test_utc_time_is_correct():
#     df = cache.copy()

#     akl = timezone("Pacific/Auckland")
#     local_dt = akl.localize(datetime(2016, 4, 1, 0, 0, 0))
#     utc_dt = local_dt.astimezone(timezone("UTC"))

#     # df = df.where(df["UTCDT"] == utc_dt.).sort_values(["TradingDate", "TradingPeriod"]).dropna()

#     df = set_time_idx(df)

#     assert df["time_idx"].min() == 0


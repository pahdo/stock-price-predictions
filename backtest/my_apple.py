from zipline.api import order_target, record, symbol
import pandas as pd
from datetime import timedelta

def initialize(context):
    context.i = 0
    context.df = pd.read_csv('../backtest/strategy.csv')
    context.row = 0
    context.start_date = pd.to_datetime('2010-1-6')

def handle_data(context, data):
    context.i += 1
    # Skip first 5 days to get full windows
    if context.i < 5:
        return
    
    # Yesterday
    trades = context.df.loc[pd.to_datetime(context.df['dates']) == context.start_date+pd.DateOffset(days=context.i-1)]
    for index, row in trades.iterrows():
        if index == 0:
            continue
        try:
            asset = symbol(row['assets'])
        except:
            #print("{} not found.".format(row['assets']))
            continue
        if row['actions'] == 1:
            order_target(asset, 0)
        elif row['actions'] == -1:
            order_target(asset, 0)

    trades = context.df.loc[pd.to_datetime(context.df['dates']) == context.start_date+pd.DateOffset(days=context.i)]
    for index, row in trades.iterrows():
        if index == 0:
            continue
        try:
            asset = symbol(row['assets'])
        except:
            #print("{} not found.".format(row['assets']))
            continue
        if row['actions'] == 1:
            order_target(asset, 100)
        elif row['actions'] == 0:
            order_target(asset, 0)
        elif row['actions'] == -1:
            order_target(asset, -100)
    record(AAPL=data.current(symbol('AAPL'), 'price'))
                
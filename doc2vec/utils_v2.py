from itertools import tee
import datetime
import formic
import glob
import os
import re
import time
import numpy as np
import sqlite3


###### CONFIGURATION ######

cik_ticker_path = os.path.join('data', 'csv', 'cikTicker.txt')
db_path = os.path.join('data', 'database', 'stocks.db')

###########################

def load_texts(directory, subset, quarters):
    """
    args :
        directory : directory for regex
        subset : 'all' or 'subset' , which matches quarters
        quarters : quarters to match e.g., [2007/QTR1]
    returns :
        generator for (text document, file path)
    """
    regex = build_regex(directory, subset, quarters)
    
    """https://bitbucket.org/aviser/formic/issues/12/support-python-3
    https://github.com/scottbelden/formic
    formic python 3 support
    formic is much much much faster than iglob
    """
    fileset = formic.FileSet(include=regex)
    for file_path in fileset.qualified_files():      
        with open(file_path, 'r') as t:
            text = t.read()
            print("loaded {}".format(file_path))
            yield (text, file_path) # file_path is a full path

def load_data(directory, subset, train_quarters, yield_paths=False):
    """generator function for dataset. streams sec forms, stock price history, and normalized returns.
    args:
        directory d: data/[d]/2004/QTR2/
        subset=['all', 'subset']
    returns:
        [[text document, price history], [alpha1, alpha2, alpha3, alpha4, alpha5]]
    """
    regex = build_regex(directory, subset, train_quarters) 
    fileset = formic.FileSet(include=regex)
    file_paths =  fileset.qualified_files()

    if(not check_file(db_path)):
         print("db_path {} does not exist.".format(db_path))
         return
    conn = sqlite3.connect(db_path)
    if not verify_db(conn):
         print("db contains too few rows.")
         return 
    cik_dict = build_cik_dict()
    if cik_dict is None:
        print("cik_dict failed")
        return

    for txt in file_paths:
        cik, filing_date = parse_txt_name(txt)
        if cik in cik_dict:
            # TODO: Query for SPY only once. 
            # Issue w/ above TODO: Must find idx of start date if we query for all SPY returns at once
            spy_returns = get_returns(conn, 'SPY', filing_date, 20) # horizon = 20
            if spy_returns is None or len(spy_returns) == 0:
                # fail silently if no SPY returns exist
                continue
            price_history, alpha1, alpha2, alpha3, alpha4, alpha5 = get_labels(conn, cik_dict[cik], filing_date, spy_returns)
            if price_history is None or len(price_history) != 5 or alpha1 is None or alpha2 is None or alpha3 is None or alpha4 is None or alpha5 is None:
                # fail silently if price history is incomplete
                continue
            with open(txt, 'r') as t:
                print("loading data/labels for {}".format(txt))
                if yield_paths:
                    yield [[t.read(), price_history], [alpha1, alpha2, alpha3, alpha4, alpha5], txt]
                else:
                    yield [[t.read(), price_history], [alpha1, alpha2, alpha3, alpha4, alpha5]]

def build_regex(directory, split, quarters):
    """ Build a regex to match annual reports
    args :
        directory : name of dir to create regex of form data/[dir]/...
        split : 'all' = ** , 'subset' = quarters
        quarters : list of quarters to match e.g., ['2007/QTR1']
    returns :
        regex : regex string , supports formic
    """
    if split == 'all':
        regex_part = '**'
    elif split == 'subset':
        regex_part = quarters
    # formic does not support '..' in a glob
    regex = os.path.join('data', directory, regex_part, '*.txt')
    print(regex)
    return regex

def verify_db(conn):
    c = conn.cursor()
    results = []
    for row in c.execute("SELECT * \
        FROM stocks WHERE symbol='AAPL' LIMIT 20"):
        results.append(row)
    #print(results)
    return len(results) == 20

def get_labels(conn, symbol, date, spy_returns, horizon=20, normalized=True):
    """get_labels
    args:
        conn: sqlite3 connection
        symbol: stock ticker symbol
        date: datetime.date object - filing date
        horizon: number of days from filing date + START_DATE_OFFSET to query returns for
                 defaults to 20. 3 day buffer for weekends + 8 from date subtract 
                 + 5 for alpha i = 5 + 3 day buffer for weekends + 1 (because upper bound is exclusive)
        normalized: defaults to True. false unsupported.
    returns:
        a tuple ([price history], alpha1, alpha2, alpha3, alpha4, alpha5)
    """
    if not normalized:
        print("get_labels: unnormalized is not supported")
        return None, None, None, None, None, None 

    stock_returns = get_returns(conn, symbol, date, horizon)
    if len(stock_returns) == 0:
        # fail silently
        # print("get_labels: get_returns query returned nothing for ticker")
        return None, None, None, None, None, None

    to_return = [get_price_history(date, stock_returns, spy_returns)]
    for i in range(1, 6):
        to_return.append(get_alpha_i(date, stock_returns, spy_returns, i))
    return tuple(to_return)

def get_returns(conn, symbol, date, horizon):
    """get_returns
    args:
        conn: sqlite3 connection
        symbol: stock ticker symbol
        date: datetime.date object - filing date
        horizon: number of days from filing date + START_DATE_OFFSET to query returns for
    returns:
        a map of datetime.date to [(datetime.date), symbol (string), return (float), beta (float)]
    """
    START_DATE_OFFSET = -8 # 3 day buffer for weeks + 5 days for baseline price history features
    c = conn.cursor()
    start_date = date_add(date, START_DATE_OFFSET)
    assert date != start_date
    stock_returns = {}
    for row in c.execute("SELECT theDate, symbol, return, beta \
        FROM stocks WHERE symbol=? \
        AND theDate >= ? \
        AND theDate < date(?, ?);",
        [symbol, start_date, start_date, '+{} day'.format(horizon)]):
        new_row = []
        new_row.append(date_add(row[0], 0))
        new_row.append(row[1])
        new_row.append(float(row[2]))
        new_row.append(float(row[3]))
        stock_returns[new_row[0]] = new_row # date addition 
    return stock_returns

def get_price_history(filing_date, stock_returns, spy_returns, days=5):
    NUM_DAYS_FOR_BUFFER = 3 # to account for non-trading days
                            # suppose days = 5, we want 5 trading days
                            # so suppose there was a national holiday
                            # and a 2-day weekend between our 5 trading
                            # days, we would want a buffer of 3 to
                            # account for that
    """assume the sec form is filed on a trading day
    """
    if filing_date not in stock_returns:
        # fail silently
        # print("get_price_history: filing date not in stock_returns map")
        return 
    beta = stock_returns[filing_date][3]
    # builds price history backwards from the filing date
    # the order doesn't matter; it's just 5 baseline features 
    current_date = date_add(filing_date, -1)
    assert current_date != filing_date
    end = date_add(filing_date, -1 * (days + NUM_DAYS_FOR_BUFFER))
    assert end != filing_date
    assert current_date != end
    days_counter = 0
    price_history = []
    while current_date != end and days_counter != days: 
        if current_date in spy_returns:
            if current_date in stock_returns:
                stock_return = stock_returns[current_date][2]
                spy_return = spy_returns[current_date][2]
                normalized_return = stock_return - beta * spy_return
                price_history.append(normalized_return)
                days_counter += 1
            else:
                print("get_price_history: missing value in stock_returns")
                return
        current_date = date_add(current_date, -1)
    if current_date == end:
        pass
        #print("get_price_history: ended because current_date == end")
    if days_counter == days:
        pass
        #print("get_price_history: ended because days_counter == days")
    if len(price_history) != days:
        # fail silently
        # print("get_price_history: missing price history values len(price_history)={} days={}".format(len(price_history), days))
        return
    return price_history

def get_alpha_i(filing_date, stock_returns, spy_returns, i):
    # TODO: Consider when an sec form is filed on a day when the market is NOT open
    """assume the sec form is filed on a day when the market is open
    """
    if filing_date not in stock_returns:
        print("get_alpha_i: filing date not in stock returns")
        return
    NUM_DAYS_FOR_BUFFER = 3
    end = date_add(filing_date, i + NUM_DAYS_FOR_BUFFER)
    current_date = filing_date 
    cum_stock_return = 1.0
    cum_spy_return = 1.0
    beta = stock_returns[current_date][3]
    days_counter = 0
    while current_date != end and days_counter != i: 
        if current_date in spy_returns:
            if current_date in stock_returns:
                cum_stock_return *= 1 + stock_returns[current_date][2]
                cum_spy_return *= 1 + spy_returns[current_date][2]
                days_counter += 1
            else:
                print("get_alpha_i: missing value in stock_returns")
                return
        current_date = date_add(current_date, 1)
    if days_counter != i:
        print("get_alpha_i: days_counter={} != i={}".format(days_counter, i))
        return None
    return (cum_stock_return - 1) - beta * (cum_spy_return - 1)

def build_cik_dict():
    """Parses cikTicker.txt file into a dictionary.
    returns:
        Dictionary object if cikTicker.txt is valid. None otherwise.
    """
    if check_file(cik_ticker_path):
        with open(cik_ticker_path) as cik_file:
            cik_ticker = cik_file.read()  
        cik_ticker = cik_ticker.replace('\n', '')
        cik_ticker = cik_ticker.replace('"', '')
        cik_ticker = cik_ticker.replace('"', '')
        cik_ticker = cik_ticker[2:-2].split('}, {')
        cik_dict = dict(s.split(', ') for s in cik_ticker)
        return cik_dict
    else:
        print('cikTicker.txt does not exist at path {}.'.format(cik_ticker_path))
        return None

def check_file(filename):
    """Check presence of a file.
    args:
        filename: path of file to check
    returns:
        bool: True for file exists and has size greater than 0. False otherwise.
    """
    try:
        if os.stat(filename).st_size > 0:
            return True
        else:
            return False
    except OSError:
        return False

def date_add(date_str, delta):
    """date addition function. 
    args:
        date_str: string or datetime.date object inputs. both are accepted.
    returns:
        datetime.date object representing date + delta
    """
    if isinstance(date_str, str):
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d') + datetime.timedelta(days=delta)
    else: # i am already a date obj
        date_obj = date_str + datetime.timedelta(days=delta)
    return date_obj

def parse_txt_name(txt):
    """parses the path name of a sec form document into its cik and filing date.
    args:
        txt: path name of sec form
    returns
        cik: key that maps to stock ticker
        date: the date when the sec form was filed (datetime.date obj)
    """
    txt = os.path.basename(txt)
    pattern = "edgar_data_(.*?)_"
    m = re.search(pattern, txt)
    if m:
        cik = m.group(1)
    pattern = "(\d{8})_"
    m = re.search(pattern, txt)
    if m:
        date = m.group(1)
    date = '{}-{}-{}'.format(date[0:4], date[4:6], date[6:])
    date = datetime.datetime.strptime(date, '%Y-%m-%d') 
    return cik, date

"""https://stackoverflow.com/questions/28030095/how-to-split-a-python-generator-of-tuples-into-2-separate-generators
"""
def split_gen(gen):
    gen_a, gen_b = tee(gen, 2)
    return (a for a, b in gen_a), (b for a, b in gen_b)

def split_gen_3(gen):
    gen_a, gen_b, gen_c = tee(gen, 3)
    return (a for a, b, c in gen_a), (b for a, b, c in gen_b), (c for a, b, c in gen_c)

def split_gen_5(gen):
    gen_a, gen_b, gen_c, gen_d, gen_e = tee(gen, 5)
    return (a for a, b, c, d, e in gen_a), (b for a, b, c, d, e in gen_b), (c for a, b, c, d, e in gen_c), (d for a, b, c, d, e in gen_d), (e for a, b, c, d, e in gen_e)

def bin_alpha(a):
    threshold = 0.01
    if a < -1 * threshold:
        return -1
    elif a > threshold:
        return 1
    else:
        return 0

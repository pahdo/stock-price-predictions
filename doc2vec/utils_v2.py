import datetime
import glob
import os
import re
import numpy as np
import sqlite3

###### CONFIGURATION ######

train_quarters = ['2005/QTR1']
"""
train_quarters = [
    '2005/QTR1', '2005/QTR2', '2005/QTR3', '2005/QTR4',
    '2004/QTR1', '2004/QTR2', '2004/QTR3', '2004/QTR4',
    '2003/QTR1', '2003/QTR2', '2003/QTR3', '2003/QTR4',
    '2002/QTR1', '2002/QTR2', '2002/QTR3', '2002/QTR4',
    '2001/QTR1', '2001/QTR2', '2001/QTR3', '2001/QTR4',
    '2000/QTR1', '2000/QTR2', '2000/QTR3', '2000/QTR4',
    '1999/QTR1', '1999/QTR2', '1999/QTR3', '1999/QTR4',
    '1998/QTR1', '1998/QTR2', '1998/QTR3', '1998/QTR4',
    '1997/QTR1', '1997/QTR2', '1997/QTR3', '1997/QTR4',
    '1996/QTR1', '1996/QTR2', '1996/QTR3', '1996/QTR4',
    '1995/QTR1', '1995/QTR2', '1995/QTR3', '1995/QTR4',
    '1994/QTR1', '1994/QTR2', '1994/QTR3', '1994/QTR4',
    '2009/QTR1', '2009/QTR2', '2009/QTR3', '2009/QTR4',
    '2008/QTR1', '2008/QTR2', '2008/QTR3', '2008/QTR4',
    '2007/QTR1', '2007/QTR2', '2007/QTR3', '2007/QTR4',
    '2006/QTR1', '2006/QTR2', '2006/QTR3', '2006/QTR4']
"""
test_quarters = [
    '2013/QTR2', '2013/QTR3', '2013/QTR4',
    '2012/QTR1', '2012/QTR2', '2012/QTR3', '2012/QTR4',
    '2011/QTR1', '2011/QTR2', '2011/QTR3', '2011/QTR4',
    '2010/QTR1', '2010/QTR2', '2010/QTR3', '2010/QTR4']
cik_ticker_path = os.path.join('..', 'data', 'csv', 'cikTicker.txt')
db_path = os.path.join('..', 'data', 'database', 'stocks.db')

###########################
"""
def load_texts(directory, split=['all', 'train', 'test'], yield_paths=False, yield_labels=False):
    regex = build_regex(directory, split)
    file_paths = glob.iglob(regex, recursive=True)
    if yield_paths:
        for file_path in file_paths:
            with open(file_path, 'r') as t:
                yield (t.read(), file_path) # file_path is a full path
    else:
        for file_path in file_paths:
            with open(file_path, 'r') as t:
                yield (t.read())
"""

def build_regex(directory, split=['all', 'train', 'test']):
    """assumes source directory structure: ../data/10-X_C/2004/QTR2/
    """
    if split == 'all':
        regex_part = '**'
    elif split == 'train':
        """https://stackoverflow.com/questions/33406313/how-to-match-any-string-from-a-list-of-strings-in-regular-expressions-in-python
        """
        regex_part = '|'.join(train_quarters)
    elif split == 'test':
        regex_part = '|'.join(test_quarters)
    regex = os.path.join('../data', directory, regex_part, '*.txt')
    #print(regex)
    return regex

def verify_db(conn):
    c = conn.cursor()
    results = []
    for row in c.execute("SELECT * \
        FROM stocks WHERE symbol='AAPL' LIMIT 20"):
        results.append(row)
    #print(results)
    return len(results) == 20
    
def load_data(directory, split=['all', 'train', 'test'], yield_paths=False, yield_labels=False):
    regex = build_regex(directory, split) 
    file_paths = glob.iglob(regex, recursive=True)

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
        # TODO: put filing_date conversion from string to date obj into parse_txt_name
        filing_date = datetime.datetime.strptime(filing_date, '%Y-%m-%d') 

        if cik in cik_dict:
            # TODO: Query for SPY only once. 
            # Challenge: How to set date range? And then how to find index of date (start date) in spy_returns?
            # default parameter for horizon = 20
            spy_returns = get_returns(conn, 'SPY', filing_date, 20)
            if spy_returns is None or len(spy_returns) == 0:
                # fail silently
                # print("no spy_returns for {} {}".format(cik_dict[cik], filing_date))
                continue
            price_history, alpha1, alpha2, alpha3, alpha4, alpha5 = get_labels(conn, cik_dict[cik], filing_date, spy_returns)
            if price_history is None or len(price_history) != 5 or alpha1 is None or alpha2 is None or alpha3 is None or alpha4 is None or alpha5 is None:
                continue
            with open(txt, 'r') as t:
                yield [t.read(), [price_history, alpha1, alpha2, alpha3, alpha4, alpha5]]

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

    Returns:
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

    Returns:
        bool: True for file exists and has size greater than 0. False otherwise.
    """
    try:
        if os.stat(filename).st_size > 0:
            return True
        else:
            return False
    except OSError:
        return False

"""date addition function. can take string or date object inputs. returns datetime.date object
"""
def date_add(date_str, delta):
    if isinstance(date_str, str):
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d') + datetime.timedelta(days=delta)
    else: # i am already a date obj
        date_obj = date_str + datetime.timedelta(days=delta)
    return date_obj

def parse_txt_name(txt):
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
    return cik, date

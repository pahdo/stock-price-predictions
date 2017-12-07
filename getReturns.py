import numpy as np

def getRets(conn, symbol, date, horizon=1):
    c = conn.cursor()
    
    results = []
    for row in c.execute("SELECT theDate, Symbol, Return, Alpha \
                         FROM stocks WHERE symbol=? \
                         AND theDate >= strftime(?) \
                         AND theDate < date(strftime(?), ?);", 
                         [symbol, date, date, '+{} day'.format(horizon)]):
        results.append(row)
    return results

def getTotalRet(conn, symbol, date, horizon=1, subtract=False):
    rets = getRets(conn, symbol, date, horizon)
    the_rets = []
    for ret in rets:
        the_rets.append(ret[2])
    return np.product([1 + float(the_ret) for the_ret in the_rets])-1
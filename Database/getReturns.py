import sqlite3

def getRets(symbol, date, horizon=1):
    conn = sqlite3.connect('../Database/stocks.db')
    c = conn.cursor()
    
    results = []
    for row in c.execute("SELECT theDate, Symbol, Return, Alpha \
                         FROM stocks WHERE symbol=? \
                         AND theDate >= strftime(?) \
                         AND theDate < date(strftime(?), ?);", 
                         [symbol, date, date, '+{} day'.format(horizon)]):
        results.append(row)
    return results
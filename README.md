## Create stocks_alpha.csv from stocks.csv
- Upload text-analytics-for-accountancy/data/stocks/stocks.csv
- Upload text-analytics-for-accountancy/data/stocks/spy.csv
- NOTE: stocks.csv does not contain data for SPY ticker
- Run text-analytics-for-accountancy/doc2vec/stock_alphas.csv

## Prepare stocks SQLite DB
- Navigate to text-analytics-for-accountancy/Database
  - `sqlite3`
  - `.mode csv`
  - `.separator ,`
  - `CREATE TABLE stocks ('theDate' int, 'symbol' text, 'return' float, 'beta' float, 'alpha' float);`
  - `.import ../stocks/stocks_alphas.csv stocks`
  - verify that import was successful: 
    - `SELECT * FROM stocks LIMIT 20;`
    - `SELECT * FROM stocks WHERE symbol='AAPL' and theDate > '2017-06-01' AND theDate < date('2017-06-01', '+5 day');`
  - `CREATE INDEX stock_index on stocks (theDate, symbol);`
  - `.save stocks.db`
  - `sqlite3 stocks.db`


## Doc2Vec Model
- Run Doc2vec/prep_data_v2.py 
- Run Doc2vec/tf_idf_v2.py

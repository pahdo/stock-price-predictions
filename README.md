## Prepare stocks .csv

- Create directory: text-analytics-for-accountancy/data/Stock_Returns/Original
- Upload stocks.csv to directory
- Open NLP-by-Alpha_Augment_Stock_Data.ipynb
  - Configure path to location data/Stock_Returns
  - Configure path_orig to location data/Stock_Returns/Original
  - Run notebook, this will create data/Stock_returns/Stocks_Plus_Alphas.csv

## Prepare stocks SQLite DB
- Check if there are column headers in Stocks_Plus_Alphas.csv. If not, add the header:
`theDate,symbol,return,empty1,empty2,empty3,empty4,empty5,alpha`
  - Open up a JupyterHub terminal window (Using the new button on the top right)
  - Navigate to data/Stock_Returns
  - vim Stocks_plus_Alphas.csv
  - Add the header
- Open terminal, navigate to text-analytics-for-accountancy/Database, and type the following commands:
  - `sqlite3`
  - `.mode csv`
  - `.separator ,`
  - `CREATE TABLE stocks ('theDate' int, 'symbol' text, 'return' float, 'empty1' text, 'empty2' text, 'empty3' text, 'empty4' text, 'empty5' text, 'alpha' float);`
  - `.import ../data/Stock_Returns/Stocks_Plus_Alphas.csv stocks`
  - `CREATE INDEX stock_index on stocks (theDate, symbol);`
  - `.save stocks.db`
- Let's test that the database was created properly. Type the following command in the terminal:
  - `sqlite3`
  - `.mode insert`
  - `.open stocks.db`
  - `SELECT * FROM stocks WHERE symbol='AAPL' and theDate > strftime('2017-06-01');`
  - Press Ctrl+C twice to exit SQLite

## Doc2Vec Model
- Run Doc2vec/group_docs_by_sentiment.py to organize documents for training
- Run Doc2vec/prep_data.py to clean and aggregate docs into files
- Run Doc2vec/doc2vec.py to train and save 3 Doc2Vec models
- Run Doc2vec/load_model.py to evaluate trained document vectors

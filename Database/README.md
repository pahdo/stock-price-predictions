## Create a SQLite3 Database of stocks data from Stocks_Plus_Alphas.csv

1. Acquire Stocks_Plus_Alphas.csv or build Stocks_Plus_Alphas.csv from stocks.csv and NLP-by-Alpha_Augment_Stock_Data.ipynb
2. Check if there are column headers in Stocks_Plus_Alphas.csv. If not, add the header:
`theDate,symbol,return,empty1,empty2,empty3,empty4,empty5,alpha`
3. Open terminal and type the following commands:
`sqlite3`
`.mode csv`
`.separator ,`
`CREATE TABLE stocks ('theDate' int, 'symbol' text, 'return' float, 'empty1' text, 'empty2' text, 'empty3' text, 'empty4' text, 'empty5' text, 'alpha' float);`
`.import Stocks_Plus_Alphas.csv stocks`
`.save stocks.db`
4. Now, to load your database, you can type the following command in the terminal:
`sqlite3`
`.mode insert`
`.open stocks.db`
5. When your database is loaded in memory, you can verify that it's working correctly with the following query:
`SELECT * FROM stocks WHERE symbol='AAPL' and theDate > strftime('2017-06-01');`
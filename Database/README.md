Get Stocks_Plus_Alphas.csv and check if there is a header. If not, add the header:
theDate,symbol,return,empty1,empty2,empty3,empty4,empty5,alpha

sqlite3

.mode csv
.separator ,

CREATE TABLE stocks ('theDate' int, 'symbol' text, 'return' float, 'empty1' text, 'empty2' text, 'empty3' text, 'empty4' text, 'empty5' text, 'alpha' float);

.import Stocks_Plus_Alphas.csv stocks

SELECT * FROM stocks WHERE symbol='AAPL' and theDate > strftime('2017-06-01');

.save stocks.db

sqlite3

.mode insert
.open stocks.db

SELECT * FROM stocks WHERE symbol='AAPL' and theDate > strftime('2017-06-01');
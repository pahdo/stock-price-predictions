### Text Mining and ML on Finance Data
#### Symposium Application
Title: On the Predictive Power of Textual Features for Stock Price Movement Predictions
Type: Poster 
Abstract: In the internet age, new unstructured text data related to public companies is constantly being released. These documents contain information that is later reflected as price movements in the stock market. We investigate the extent to which text data can inform predictions about stock price movements. We develop a system to learn from financial text data of a specific type: SEC Form 10-Ks. We predict the direction of stock price movements (negative, neutral, positive) normalized to the broader stock market for up to 5 days after the release of the text document. To do so, we use modern machine learning techniques such as gradient boosting and deep learning for document embeddings. We compare our model that leverages textual features with a baseline model that uses only stock price history. Our use of textual features is found to considerably improve predictive performance of future stock price movements.
#### Project Outline
Data
 * 10-Ks
 * stock returns
Pre-processing
 * lemmatization
 * stop word removal
 * non-dictionary words removal
 * TODO: words between a negation and the end of a sentence are suffixed with "--n"
 * stock return normalization with SPY
 * stock return binning into NEGATIVE, NEUTRAL, POSITIVE
Features
 * baseline
 * linguistic features from 10-Ks data set
Labels
 * 1-day normalized returns
 * 2-day normalized returns
 * 3-day normalized returns
 * 4-day normalized returns
 * 5-day normalized returns
   * tests predictive power of linguistic features across various time frames
Models
 * baseline: 5-day price history
 * model 1: baseline + NMF(TF-IDF)
   * non-negative matrix factorization of tf-idf vectors
 * model 2: baseline + Doc2Vec
Results
 * grid parameter search with time-series cross validation

Additional Notes
Differences from the LREC paper
  * the models I'll use (above) are more modern than those in the LREC paper
  * LREC paper does something similar, the differences are: data set (I use 10-Ks instead of 8-Ks), models/algorithms, and baseline (I use price history instead of 21 quantitative features like earnings, etc.)
Results from the LREC paper
  * linguistic features only add 4.23% accuracy to the baseline
  * benefit decreases with time (less predictive power from 1-day to 5-day)
  * I can compare my results to the LREC paper in my thesis
  * TODO: Is this reasonable? Different data are used
Differences from last semester
  * time series cross-validation instead of k-fold cross-validation
  * testing against a better baseline than random chance
  * I can compare my results to the LREC paper
  * greater code quality
  * lemmatization and negations in the text preprocessing step
  * dimensionality reduction with TF-IDF
Primary References
  * Lee et al. 2014 in LREC journal: https://nlp.stanford.edu/pubs/lrec2014-stock.pdf
  * Peng et al. 2015 on arXiv: https://arxiv.org/pdf/1506.07220.pdf
#### Other
Introduction (draft): In the internet age, unstructured text data related to public companies is constantly being released. Examples include tweets, news articles, earnings call transcripts, and SEC forms. The information in these documents is slowly digested by humans and the market eventually reflects this new information. However, using an automatic system to predict stock price movements following the release of a text document may provide traders with an edge.
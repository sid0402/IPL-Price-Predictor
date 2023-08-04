# IPL-Price-Predictor
A machine learning project to determine cricket player prices in the Indian Premier League annual auctions. The dataset has been built by scraping data from the howstat website. A Support Vector Regression (SVR) model has been developed to predict prices for batsman and wicketkeepers, and a Linear Regression model has been built to predict prices for bowlers and all-rounders. The entire project has been built using Python and its libraries.

# Installation
The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using:

```
pip install -r requirements.txt
```

# Usage
To build the dataset "AuctionId.csv", run the following (in the specified order) on terminal:
```
python scrapers/id_scraper.py
python scrapers/stats_scraper.py
python scrapers/stats_analyzer.py
```

To process the dataset, run the following on terminal:
```
python ml/data_processing.py
```

The two files, ml/train.py and ml/train_bowl.py builds the batting/wicketkeeper model and bowler/allrounder models respectively.

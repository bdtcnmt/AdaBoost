# Project 2 - AdaBoost

## Overview
This project uses sentiment analysis and AdaBoost to predict daily stock market movements (bullish vs. bearish) 
based on news headlines aggregated from a Kaggle dataset. The program converts the 25 daily headlines into a 
comprehensive representation using two complementary techniques: TF-IDF, which extracts detailed word-level 
features by weighting words according to their frequency relative to the overall corpus, and sentiment analysis, 
which computes an overall emotional tone for the day's headlines using the VADER lexicon. This combined 
representation is then used to train an AdaBoost ensemble, which iteratively build a strong classifier by 
combining multiple weak learners--specifically, shallow decision trees. Each weak learner is trained on reweighted 
data that emphasizes previous misclassifications, thereby focusing on the most difficult examples. The finals model 
aggregates these weighted predictions to improve the overall predictive performance. This project demonstrates how 
AdaBoost improves upon the performance of a single weak learner.

## Directory Structure
```plaintext
CAP4601_project_2/
├── data/
│   └── Combined_News_DJIA.csv
├── src/
│   ├── adaboost.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── main.py
│   ├── visualization.py
│   └── __init__.py
├── README.md
├── Report.pdf
└── requirements.txt
```

## How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Navigate to src/ folder

3. Run main
```bash
python main.py
```
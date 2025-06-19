
# Analysis of price of apartment for polish cities and ML-model training

## Overview

The goal of the project is to analyse factors contribution on price of the rent, and purchase of apartment for Warsaw, Wrocław, Lublin, Radom, and Łódz. The data were used to training machine learning model, which predicts value of rent in Wrocław, in dependency on such factors like: distance from center, area, year of the builing, bulining type, material of buliding, flor number, total number of floors, number of rooms, and avability of parking space, balcony, storage room, elevator, or security.

### Analysis of

- Price and distance to center distribution for selected cities
- Dependence of price and presence parking space, balcony, storage room, elevator, or security.
- Statistical analysis for these binary features (Mann-Whteney U, T-test, Kolmogorov-Smirnov Test, Cliff's Delta).
- Average price for rent and purchase of apartment between August 2023 and June 2024 for selected cities.
- Machine Learning models, and their parameters.
- Machine Learning models' accuracy and efficiency (training and fitting time measurement, MAE, R$^2$, cross validation, and RMSE calculations).

### ML-based price prediction app for Wrocław

## Project Structure

bash```

├─data/               # Datasets (original, intermediate, cleaned)
├─ML_models/          # Machine Learning models for Linear Reggresion, Random Forest, and Extreme Gradient Boosting (XGB)
│
├─plots/              # Plots and charts for price and distance distribution, average price for rent or purchase for each city over time, and features importance for ML
│   │
│   └─data_two_sides/ # Plots of comparison for binary features distribution
│
├─reports/            # Statistical analysis for binary features, its easy to compare equlivalent table, and machine learning models comparison of efficiency and accuracy.
│
└─scripts/
        1_data_overview.ipynb                    # Data overview: city with the largest number of records, cities spectrum
        2_data_cleaning.ipynb                    # Pre-data cleaning, summary statistics of datasets
        3_data_analysis_statistics_intro.ipynb   # Pre-statiscics for binary features, price distribution for aparments rent and purchase in dependency of avalibilty of elevator
        4_statistical_analysis.ipynb             # Statistical analysis of binary features, analysis of their correlation witch price. Selection of feature with the higest correlation, and generation of plots of distribution.
        5_average_price_vs_time.ipynb            # Analysis of average price for rent and purchase of apartment over time.
        6_centre_distance.ipynb                  # Analysis of price and distance distrubution for selected cities.
        7_ML_testing_wroclaw.ipynb               # Features selection for ML models, evaluation models parameters, and comparison ML models.
        8_best_ML_model_selection.ipynb          # Comparison of accuracy, and efficiency of ML models.                             9_price_prediction.ipynb                 # ML model pipeline generation for Wrocław rent price prediction.
        10_app.py                                # Script to predict price of rent in Wrocław by browser survey.
        my.py                                    # Functions.
```

# Data Description

Source: https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland

Features: city, area of flat, type of builiding, builidning material, age of buliding, ownership, condition, poi, number of rooms, floor, total number of floors in buliding, latitude, longitude distance to: center, universty, kindergarden, pharmacy, clinic, post office or restaurant, avaliability of parking space, balcony, storageroom, security and elevator.

Period: between August 2023 and June 2024

# Methodology

1. Data cleaning:

- NaN in hasStorageRoom, hasSecurity, has Elevator, hasBalcony, and hasParkingSpace the NaN filled with "No".

- NaN in distances filled with centereDistance.

- NaN in condition filled with "Low".

- NaN in floor filled with FloorCount.

- NaN in floorCount is filled with 6.

- NaN in bulidYear filled with random number from 1850 to 1930.

- NaN in type filled with "NoType".

- NaN in buidingMaterial filled with "concreteSlab".

2. Joining all subset for rent and purchase into one dataset.

3. Exploration of statistial methods (Mann-Whteney U, T-test, Kolmogorov-Smirnov Test, Cliff's Delta).

4. Comparison of distribution of price, and distance to center for each city, and among them.

5. Testing ML models.

6. Evaluation of time and accuraty of prediction for ML models.

7. Pipeline performance for Wrocław, and selected features for app use.

# Key Findings

# How to Run

## Clone the repo

git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/eda_modeling.ipynb

# Future Work

- Obtaing data from advertising pages for 3 other cities for time of the year

- Data analysis of price changes over time, and distance to center

# Contact


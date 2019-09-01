# Using-News-to-Predict-Stock-Movements
Sentimental analysis on news data is mapped to rolling window of stocks to predict stock movements

This project worked with the dataset from 2sigma Kaggle Competition. This project dealt with the question - Can we use the content of news analytics to predict stock price performance? 

Here are the steps we took in coming up with a model for our problem.

#### 1. Feature Engineering

1. Merge Market & News Data

2. Impute returns data using NOCB https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4

3. Explore companies that have data for news variables such as VolumeCounts7D/NoveltyCount7D. 

3.1 Explore the time series data to see which subset of data is more relavent to current problem.
  
4. Find outliers of all variables and treat them. https://www.kaggle.com/artgor/eda-feature-engineering-and-everything

5. Create new features to capture autocorrelation:  We created some stock related features that traders use, such as : RSI, Bolliger Band etc. We also fed temporal (Day, Month, Quarter etc.) features, rolling sums of News variables (News Urgency, news relavence) as they appeared more predictive than using them in the day's context.


Create new Features to account for Time Series auto Correlation between rows.

#### 2. Data reduction & Exploration

1. Subset of Data for top companies that always appear in news. We considered top 15 companies with most news data based on our research.

2. The reason for doing so was due to the abundant news articles as well as data available for those in particular.

#### EDA - Closing Prices Trends

![Alt text](ClosingPricesTrends.png?raw=true "ClosingPricesTrends.png")

#### EDA - Interpolation Of Missing Values

![Alt text](InterpolationOfMissingValues.png?raw=true "InterpolationOfMissingValues.png")

#### 3. Split train and Test

1. Transform target variable to binary Stock-Movement Up/Down (0/1)
2. Stock-Movement Up/Down will be the label for training.
 
#### 4. Compare various Classifiers to find the best one. https://www.kaggle.com/aldemuro/comparing-ml-algorithms-train-accuracy-90 

#### 5. Fit Classifier(s) with Training data using Classifiers that work well with Mixed Data.

     1. Random Forest
     2. BaggingClassifier with DecisionTrees
     3. XGBoost
      
#### 6. Cross validation to estimate test error for this model.


#### 7. Use GridSearchCV to tune hyper parameters.
Use randomSearchCV instead of GridSearchCv for faster results.

#### 8. Use the best estimator for test prediction and accuracy.


# Authors

Anupriya Thirumurthy, Madhavi Polisetty, Desen Liu


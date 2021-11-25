# Churn analysis using Spark

Churn analysis (also called customer attrition rate) is essential for companies to evaluate their customer loss rate. For any organisation, retaining the existing customers, attracting new customers, and minimising the churn is critical to maximising revenue. 

## Problem Introduction

When it comes to streaming services (music or video), the most common reasons users cancel their subscriptions are lack of content, limited customer assistance, rival services, etc. In such circumstances, organisations can run a predictive churn analysis using the available user behavioural data to accurately identify and execute necessary strategic actions to prevent users from leaving the platform. 

## Strategy to solve the problem

In this analysis, we use a large dataset from a music streaming platform provided by Udacity. Analysing large data sets and training machine learning models require distributed systems to handle extensive memory-intensive processes. Spark is one such tool designed to achieve this. Spark has built-in modules capable of pre-processing and modelling big-data using multiple nodes in a small cluster. In this project we use a local Spark installation, thus uses a small subset of the original 12 GB dataset. We test four clustering models to predict the user churn. The workflow of the is as follows,

 1. Explore, understand dataset and define churn
 2. Data pre-processing: Drop empty users and extract features
 3. Create machine learning models: Vectorise data and define core routines for classifiers
 4. Evaluation, validation and hyperparameter tuning    

## Metrics

The performance of the machine learning model can be measured using a number of performance metrics such as accuracy, precision, recall and f1-score etc. Among all these, f1-score provides a better score combining precision and recall. The f1-score can be interpreted as a harmonic mean of the precision and recall, where an f1-score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the f1-score are equal. The formula for the f1-score is:

$F_{1}=2 \cdot \frac{\text { precision } \cdot \text { recall }}{\text { precision }+\text { recall }}=\frac{\mathrm{TP}}{\mathrm{TP}+\frac{1}{2}(\mathrm{FP}+\mathrm{FN})}$

where TP = number of true positives, FP = number of false positives and FN = number of false negatives. 

## Exploratory data analysis

The provided data set contain:

 - userId
 - artist
 - auth: login status (Logged Out, Cancelled, Guest, Logged In)
 - firstName
 - gender (M / F)
 - ItemInSession: operation sequence number of this session, from small to large according to time
 - lastName
 - length: the length of the song
 - level: user tier (Free / Paid)
 - location: the user location
 - method: method of getting web pages (PUT/GET)
 - page: page(s) user visited (About, Add Friend, Add to Playlist, Cancel, Cancellation Confirmation, Downgrade, Error, Help, Home, Logout, NextSong, Roll Advert, Save Settings, Settings, Submit Downgrade, Submit Upgrade, Thumbs Down, Thumbs Up, Upgrade)
 - registration: time stamp for registration point in time
 - sessionId: user session id (determine a single login operation)
 - song
 - status: page return code (200,307,404)
 - ts: timestamp of the log time
 - userAgent: web-scraping client

The calculated user churn data is as follows:

```
+------+---------+-----+
|gender|userChurn|count|
+------+---------+-----+
|     M|        1|   32|
|     F|        0|  104|
|     F|        1|   20|
|     M|        0|  121|
+------+---------+-----+
```

The data set shows that the user churn is higher in males than in females. However, this dataset is a small subset of the larger dataset. Therefore, it is noteworthy that using the entire dataset (which would be challenging without a distributed system) will significantly improve the statistics/model performance. 

## Modelling

The original dataset needs to be processed before creating the labelled dataset along with the features vector. The following steps have been performed using PySpark:

 - drop data points with empty `userId` or `sessionId`
 - extract unique `userId`s
 - convert `ts` (timestamp) to 24H format and `datetime` format
 - extract user churn by looking at the `Cancellation Confirmation` in the `page` column
 - create a new column to store user churn `userChurn`
 - one hot encoding for `gender` (male or female) and `level` (paid or free)
 - store the number of songs each user listened to in `nSongs`
 - store the number of thumb ups, thumb downs and rolled advertisements in `thumbsUp`, `thumbsDown` and `rollAdvert` columns.

The final data table looks like this:

```
+------+------+--------+----------+----------+-------+-------+---------+---------+-----------------+---------+
|userId|nSongs|thumbsUp|thumbsDown|rollAdvert|genderF|genderM|levelFree|levelPaid|      avgSessions|userChurn|
+------+------+--------+----------+----------+-------+-------+---------+---------+-----------------+---------+
|100010|   275|      17|         5|        52|      1|      0|        1|        0|           9269.0|        0|
|200002|   387|      21|         6|         7|      0|      1|        1|        0|          15984.0|        0|
|   125|     8|       0|         0|         1|      0|      1|        1|        0|           1774.0|        1|
|    51|  2111|     100|        21|         0|      0|      1|        0|        1|          52341.4|        1|
|   124|  4079|     171|        41|         4|      1|      0|        0|        1|34739.65517241379|        0|
+------+------+--------+----------+----------+-------+-------+---------+---------+-----------------+---------+
only showing the top 5 rows
```

Then, these feature set needs to be standardised and converted into a labelled feature vector. The final data set looks like this:

```
+-----+--------------------+
|label|            features|
+-----+--------------------+
|  0.0|[0.24887761207221...|
|  0.0|[0.35023867589799...|
|  1.0|[0.00724007598755...|
|  1.0|[1.91047505121618...|
|  0.0|[3.69153374415482...|
+-----+--------------------+
only showing the top 5 rows
```

The data set needs to be split into test (20%) and training (20%) data. Then, we can train our models and evaluate the model performance. For this analysis, we use four machine learning models and f1 score as a performance measure.

Classifiers we used:

 - `DecisionTreeClassifier`
 - `LogisticRegression`
 - `RandomForestClassifier`
 - `GBTClassifier`


## Results

We have observed the default classifier parameters above 0.9 f1 scores for `DecisionTreeClassifier` and `GBTClassifier`. The resulting f1 scores are as follows;

| classifier               | f1-score (train) | f1-score (test) |
| ------------------------ | ---------------- | --------------- |
| `DecisionTreeClassifier` | 0.90             | 0.68            |
| `LogisticRegression`     | 0.67             | 0.63            |
| `RandomForestClassifier` | 0.78             | 0.63            |
| `GBTClassifier`          | 0.99             | 0.72            |

Interestingly, even with a minimum number of features and without hyperparameter optimisation, `GBTClassifier` performs very well. According to the test performance, low f1-scores indicates an over-fitting in the model training. Using a stratified sample for the test and training may avoid this, or using the whole 12GB dataset would improve the overall statistics.  

## Hyperparameter tuning

Some models performed very well even with the default parameters. However, to improve the model performance hyperparameter tuning is essential; especially for the models which under performed. Spark has the in-built functionality to perform cross validation and parameter grid creation. Using that, we have looked at a range of parameters. Most of the models showed slight or inconsiderable improvements. However, among all the models, `RandomForestClassifier` had the best performance gain from 0.78 to 0.94 f1-score. 

## Justification

Even though some classifiers resulted in a good performance, a better perfomrance can be gained through hyperparameter tuning . In this project, `RandomForestClassifier` showed a good performance gain with proper parameters. However, the process of hyperparameter tuning is time consuming and computationally expensive. 

As a performance metric to evaluate the model and hyperparameter tuning we use f1-score over other performance metrics. The harmonic mean of precision and recall is biased neither to precision nor recall. Thus, it reflects a great balance of both.

## Conclusion

According to the above discussion, the concluding reflections and recommendations for further improvements are as follows. 

### reflections

 - Overall, `GBTClassifier` and `DecisionTreeClassifier` show excellent performance, even with default classifier parameters
 - The `RandomForestClassifier` model show a considerable performance gain after hyperparameter tuning
 - All models tend to be over-fitted. Therefore, stratified sampling and/or training on a more extensive data set is needed
 - Hyperparameter tuning takes longer than expected. It requires a considerable computational time

### Improvements

The classification models we tested seems to be over-fitted since the train score is very high and  the test score is comparatively low. This requires further investigation. To avoid over-fitting and to improve the model, one could use the full 12 GB dataset and stratified sampling. Introducing more features would also help to improve the model.  

## Code availability

All code used in this project can be found here <https://github.com/madhurangar/spark-churn-prediction>

References:
 [1] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
 [2] https://spark.apache.org/docs/latest/ml-classification-regression.html 



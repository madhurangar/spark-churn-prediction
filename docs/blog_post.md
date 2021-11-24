# Churn analysis using Spark

Churn analysis (also called customer attrition rate) is essential for companies to evaluate their customer loss rate. For any organisation, retaining the existing customers, attracting new customers, and minimising the churn is critical to maximising revenue. 

When it comes to streaming services, the most common reasons users cancel their subscriptions are lack of content, limited customer assistance, rival services, etc. In such circumstances, organisations can run a predictive churn analysis using the available behavioural data to accurately identify and execute necessary strategic actions before users leave the platform. 

In this analysis, we use a large dataset from a music streaming platform provided by Udacity. Analysing large data sets and training machine learning models requires distributed systems to handle extensive memory-intensive processes. Spark is one such tool designed to achieve this.  

## The data set

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

The data set shows that the user churn is higher in males than in females. However, this data is a small subset of the larger dataset. Therefore, it is noteworthy that using the entire dataset (which would be challenging without a distributed system) will significantly improve the statistics/model performance. 

## Feature Engineering

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

## Modelling

The data set needs to be split into test (20%) and training (20%) data. Then, we can train our models and evaluate the model performance. For this analysis, we use four machine learning models and f1 score as a performance measure.

Classifiers we used:

 - `DecisionTreeClassifier`
 - `LogisticRegression`
 - `RandomForestClassifier`
 - `GBTClassifier`

We have observed the default classifier parameters above 0.9 f1 scores for `DecisionTreeClassifier` and `GBTClassifier`. The resulting f1 scores are as follows;

| classifier               | f1-score (train) | f1-score (test) |
| ------------------------ | ---------------- | --------------- |
| `DecisionTreeClassifier` | 0.90             | 0.68            |
| `LogisticRegression`     | 0.67             | 0.63            |
| `RandomForestClassifier` | 0.78             | 0.63            |
| `GBTClassifier`          | 0.99             | 0.72            |

Interestingly, even with a minimum number of features and without hyperparameter optimisation, `GBTClassifier` performs very well. According to the test performance, low f1-scores indicate an over-fitting in the model training. Using a stratified sample for the test and training may avoid this, or using the whole 12GB dataset would improve the overall statistics.  

## Remarks
 - Overall, `GBTClassifier` and `DecisionTreeClassifier` show excellent performance, even with a small subset of data. 
 - All models tend to over-fit. Therefore, stratified sampling and/or training on a more extensive data set is needed
 - Hyperparameter tuning takes longer than expected. It requires a considerable computational time
 - Finally, the project was an excellent exercise to handle Spark skillfully

## Code availability

All code used in this project can be found here <https://github.com/madhurangar/spark-churn-prediction>

# Sparkify Churn Prediction (Data Scientist Capstone, Co op Internship)

A churn prediction project for Sparkify, a fictional music streaming platform modeled after Spotify. The entire pipeline is built with Apache Spark (PySpark) and runs on Databricks. It processes raw user event logs, aggregates them into a per user feature table, then trains and compares binary classification models.

This is work I actually performed during my internship, with a supervisor reviewing and validating every step. This README documents the exact process I followed, along with the real results printed inside the notebook.

## 1. Problem Context

Sparkify records every user action as an event log: playing a song, clicking Thumbs Up, adding a friend, upgrading or downgrading a plan, and most importantly, cancelling the service.

Goal: identify users who are likely to leave early enough that the business team can offer retention incentives in time. Because the churn group is a small minority, the optimization metric is F1 Score rather than Accuracy alone.

## 2. The Data

Data file: `data/mini_sparkify_event_data.json` (about 128 MB, tracked with Git LFS as configured in `.gitattributes`). This is a subset of the full 12 GB dataset.

Actual statistics after loading:

* Log rows: 286,500
* Time range: from 2018/10/01 00:01:57 to 2018/12/03 01:11:16
* Distinct users after cleaning: 225

The schema has 18 fields: `artist`, `auth`, `firstName`, `gender`, `itemInSession`, `lastName`, `length`, `level`, `location`, `method`, `page`, `registration`, `sessionId`, `song`, `status`, `ts`, `userAgent`, `userId`.

## 3. Environment and Libraries

* Apache Spark with PySpark (`pyspark.sql`, `pyspark.ml`, `pyspark.mllib`)
* Databricks Notebook (the notebook metadata confirms a Databricks runtime, and the built in `display()` command is used throughout)
* pandas and numpy for the small scale time analysis
* matplotlib for charts

Main notebook: `FinalProjectStarted.ipynb`.

## 4. The Process, Step by Step

### Step 1: Load the data and inspect the schema

The JSON data is read directly with `spark.read.json()`. I printed the schema to confirm the type of every column, in particular that `ts` and `registration` are epoch milliseconds stored as long, and that `userId` is a string.

### Step 2: Establish the time boundaries of the dataset

I used `to_timestamp(col('ts')/1000)` together with `min` and `max` to confirm the data spans roughly two months. This step matters because it tells me the range the `lifetime` feature can take, and exactly which time window the model is learning from.

### Step 3: Define the Churn label

This is the pivotal step of the project. The logic has two parts:

1. Create a UDF that flags 1 for every log row where `page == "Cancellation Confirmation"`, and 0 otherwise. This event applies to both free and paid users, so it accurately captures leaving behavior.
2. Propagate the label across the user's entire history using a Window Function. I used `Window.partitionBy("userId").rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)` followed by `Fsum` on the churn column. Result: if a user ever cancelled, every log row belonging to that user carries the label 1.

Why the propagation step is necessary: if only the single cancellation row were flagged, the label would lose its meaning the moment I group by user to build features. The Window Function solves this without an extra join, and keeps the computation distributed across Spark.

### Step 4: Check and handle missing data

I wrote a `missing_values()` helper that checks three conditions at once: `isnan`, `isNull`, and the empty string. Actual results:

* `artist`, `length`, `song` are missing on 58,392 rows
* `firstName`, `gender`, `lastName`, `location`, `registration`, `userAgent`, `userId` are missing on 8,346 rows

Interpretation: the 58,392 rows missing song information are simply events that are not song plays (for example Home, Settings, Logout). That is not a data defect, so I kept them. In contrast, the 8,346 rows with an empty `userId` belong to sessions from users who were not logged in. They cannot be labeled and cannot contribute to per user features, so I dropped them with `df.filter(df["userId"] != "")`.

Rows remaining after cleaning: 278,154.

### Step 5: Exploratory Data Analysis, quantitative view

I took one representative record per user with `dropDuplicates(['userId'])` so that heavy users would not be counted repeatedly when computing rates.

Actual results:

**By gender**

* Churn rate among female users: 19.23 percent
* Churn rate among male users: 26.45 percent

**By subscription tier**

* Churn rate on the free tier: 24.86 percent
* Churn rate on the paid tier: 16.67 percent

Observation: paid users are substantially more loyal than free users, which matches business expectations.

**By geography**

The `location` column has the form "City, State", so I extracted the state with `split(location, ',').getItem(1)` and then aggregated churn counts by state. Top five states: CA (6), NY NJ PA (5), MI (3), FL (3), TX (3).

**By artist**

I aggregated churn by `artist` to see which artists appear most often in the history of users who left. This was a reference check only. I did not feed it into the model because it is noisy and extremely high in cardinality.

### Step 6: Time based analysis

I created four UDFs to extract `hour`, `day`, `month`, and `week_day` from the `ts` field. I then converted to pandas with `toPandas()` for plotting, which is safe here because the dataset is already small enough.

The `get_series()` function counts activity at each time bucket, split into the churn and non churn groups, with an option to normalize into percentages. The `draw_time()` function then plots a bar chart comparing the two groups.

Normalizing to percentages is mandatory here: the churn group is a minority, so plotting absolute counts would make the two groups visually incomparable.

### Step 7: Feature Engineering

From row level event logs, I aggregated ten features at the user level. Each feature was built as its own DataFrame before being joined, which makes them easy to verify individually and easy to reuse when moving to the 12 GB dataset.

1. `lifetime`: tenure, computed as `max(ts − registration)` converted into days. Mean 79.85 days, minimum 0.31 days, maximum 256.38 days.
2. `total_songs`: total event count per user. Mean 1,236.24, maximum 9,632.
3. `num_thumb_up`: number of Thumbs Up clicks. Mean 57.05 across the 220 users who performed this action.
4. `num_thumb_down`: number of Thumbs Down clicks. Mean 12.54 across 203 users.
5. `add_to_playlist`: number of songs added to a playlist. Mean 30.35 across 215 users.
6. `add_friend`: number of friends added, a proxy for social engagement. Mean 20.76 across 206 users.
7. `listen_time`: total listening duration, computed as `sum(length)`. Mean 252,558 seconds.
8. `avg_songs_played`: average songs per session, computed by counting `NextSong` events per (`userId`, `sessionId`) pair and then averaging per user.
9. `gender`: M encoded as 0 and F as 1, cast to integer.
10. `artist_count`: number of distinct artists a user listened to, computed with `dropDuplicates` on the (`userId`, `artist`) pair followed by a count.

Dependent variable: the `churn` column renamed to `label` and deduplicated per user.

### Step 8: Assemble the training dataset

The ten feature tables and the label table are joined on `userID` with an `outer join`. The `userID` column is then dropped and missing values are filled with `fillna(0)`.

Why an outer join rather than an inner join: an inner join would drop every user who never clicked Thumbs Down or never added a friend. But "never did that at all" is itself a strong churn signal. So I keep every user and fill with 0, treating the missing value as meaning "this behavior never occurred".

### Step 9: Feature scaling

Two sequential steps:

1. `VectorAssembler` combines the ten numeric columns into a single vector column named `NumFeatures`, because Spark MLlib requires vector input.
2. `StandardScaler` with `withStd=True` brings the features onto a comparable scale, producing the `features` column.

Scaling is mandatory here because the features differ enormously in magnitude. For example `listen_time` reaches the hundreds of thousands while `gender` is only 0 or 1. Without scaling, Logistic Regression would be dominated by the few features with large absolute values and the rest would effectively stop contributing.

### Step 10: Split the dataset

Two successive splits using `randomSplit` with `seed=42` to guarantee reproducibility:

* Train: 70 percent
* Validation: about 18 percent (60 percent of the remaining 30 percent)
* Test: about 12 percent

The validation set is used to compare and select models. The test set is touched exactly once, at the very end, so the reported result stays honest.

### Step 11: Baseline models as a reference point

Before training anything, I built two naive models on the test set by hard coding the `prediction` column:

* Predict everything as 0 (nobody churns): Accuracy 0.75 and F1 Score 0.6429
* Predict everything as 1 (everybody churns): Accuracy 0.25 and F1 Score 0.1

Methodologically I consider this the single most important step. It shows that blindly guessing "nobody churns" already reaches 75 percent Accuracy, purely because of class imbalance. Every model afterwards therefore has to beat the F1 mark of 0.6429 before it can claim any real value.

### Step 12: Logistic Regression with Cross Validation

Configuration: `LogisticRegression(maxIter=10)`, evaluated with `MulticlassClassificationEvaluator(metricName='f1')`, wrapped in a `CrossValidator` with `numFolds=3` and an empty param grid to establish a baseline.

Validation set result: Accuracy 0.70 and F1 Score 0.6048.

### Step 13: Gradient Boosted Trees with Cross Validation

Configuration: `GBTClassifier(maxIter=5, seed=42)`, wrapped in a `CrossValidator` with `numFolds=5`.

Validation set result: Accuracy 0.60 and F1 Score 0.5682.

Observation: GBT usually outperforms Logistic Regression on tabular data, but here the training set holds only about 150 users, so the more complex model overfits. This result reinforced the decision to move forward with Logistic Regression.

### Step 14: Hyperparameter tuning and final evaluation

I selected Logistic Regression and built a parameter grid with `ParamGridBuilder`:

* `maxIter`: 10 and 12
* `regParam`: 0 and 0.1
* `elasticNetParam`: 0.001 and 0.01

That is 8 combinations in total, run through a `CrossValidator` with `numFolds=3` and optimized on F1. The winning model is retrieved via `bestModel` and evaluated exactly once on the test set.

Final test set results:

* Accuracy: 0.78125
* F1 Score: 0.7419

Compared to the baseline (F1 0.6429), the final model improves F1 by roughly 0.099 points, an increase of about 15 percent.

### Step 15: Feature importance analysis

For Logistic Regression, I used the absolute value of the `bestModel.coefficients` as the influence measure, then plotted them as a bar chart with matplotlib.

Resulting ranking:

1. Friends added: 2.3281
2. Tenure: 0.8869
3. Thumbs Down clicks: 0.7113
4. Total songs played: 0.5263
5. Distinct artists: 0.2956
6. Playlist additions: 0.2802
7. Thumbs Up clicks: 0.2638
8. Total listening time: 0.2251
9. Average songs per session: 0.1001
10. Gender: 0.0221

Business interpretation:

* Social behavior (adding friends) is by far the strongest signal, well ahead of pure content consumption metrics. Users with connections inside the platform are much harder to lose.
* Thumbs Down carries more weight than Thumbs Up, meaning negative signals predict churn better than positive ones.
* Gender has almost no predictive value and could be dropped in a future iteration to simplify the model.

## 5. Results Summary

Validation set, baseline Logistic Regression: Accuracy 0.70 and F1 Score 0.6048.
Validation set, Gradient Boosted Trees: Accuracy 0.60 and F1 Score 0.5682.
Test set, tuned Logistic Regression: Accuracy 0.78125 and F1 Score 0.7419.
Baseline predicting all zeros on the test set: Accuracy 0.75 and F1 Score 0.6429.

Winning model: regularized Logistic Regression, tuned through three fold Cross Validation.

## 6. How to Reproduce

1. Prepare an environment with Apache Spark. I ran this on Databricks; to run locally you need `pyspark`, `pandas`, `numpy`, `matplotlib`, and `scipy`.
2. Fetch the data. The repo uses Git LFS, so run `git lfs install` followed by `git lfs pull` to download the full `data/mini_sparkify_event_data.json`.
3. Open `FinalProjectStarted.ipynb`.
4. Adjust the path in the data loading cell to match your environment. The notebook currently uses `spark.read.json('/mini_sparkify_event_data.json')`, which follows the Databricks DBFS layout; running locally, change it to `data/mini_sparkify_event_data.json`.
5. Outside Databricks, replace the `display(...)` calls with `.show()` or `.toPandas()`, since `display` is a Databricks specific helper.
6. Run the cells sequentially from top to bottom.

## 7. Known Issues When Rerunning

* The `VectorAssembler` cell only runs correctly once, in order. In the saved notebook this cell still carries an `IllegalArgumentException` caused by rerunning it after the `data` variable had already been overwritten during the join step. If you hit the same error, rerun from the feature assembly cell (Step 8) and continue from there.
* The `data` variable is reused for different purposes throughout the notebook (first the event log enriched with time columns, later the per user feature table). This is something I would split into clearly named variables in the next revision.
* The UDFs relying on `datetime.fromtimestamp` depend on the local timezone of the machine running them. Running in a different timezone may shift the hourly charts.

## 8. Limitations and Next Steps

* Small dataset. With only 225 users, the test set holds roughly 32 users, so the metrics fluctuate heavily. The next step is running the full pipeline on the complete 12 GB dataset with a Spark cluster in the cloud.
* Class imbalance is not addressed directly. Worth trying `weightCol` in Logistic Regression, or resampling the minority class.
* Downgrade events are unused. Downgrading a plan is an early warning signal and could become either an additional feature or a secondary label.
* No trend based features yet. For example the drop in activity over the final two weeks relative to a user's historical average is typically a very strong churn signal.
* The model is not yet exposed as a service. A natural next step is persisting the Spark ML Pipeline and deploying it for scheduled batch scoring.

## 9. Repository Structure

```
Data_Scientist_Capstone_Co-op_Sparkify/
├── FinalProjectStarted.ipynb           Main notebook, the full pipeline
├── data/
│   └── mini_sparkify_event_data.json   Event logs, 128 MB, tracked via Git LFS
├── .gitattributes                      Git LFS configuration for the data file
└── README.md
```

## 10. Acknowledgements

The problem statement and dataset come from the Udacity Data Scientist Nanodegree Capstone Project. The implementation, analysis, and interpretation in this repository are my own work, produced during my internship and reviewed and validated by my supervisor.

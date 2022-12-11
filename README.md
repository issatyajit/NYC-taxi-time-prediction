# NYC-taxi-time-prediction
## **Abstract**:
New York City yellow cabs collect information regarding the trips made. One such data collected by them is the trip duration, our aim here is to make a supervised Machine Learning model which can predict the time taken by a cab.

## **Introduction**:
We are given a data set containing information on NYC yellow cab services for the year 2016. We don’t have any information regarding the customers.
The given dataset contains 1458644 trip records and 11 variables.
The variables are as follows:

id- a unique identifier for each trip

vendor_id- a code indicating the provider associated with the trip record

pickup_datetime- date and time when the meter was engaged.

dropoff_datetime- date and time when the meter was disengaged.

passenger_count- the number of passengers in the vehicle.

pickup_longitude - the longitude where the meter was engaged.

pickup_latitude - the latitude where the meter was engaged.

sropoff_longitude- the longitude where the meter was disengaged.

dropoff_latitude - the latitude where the meter was disengaged.

store_and_fwd_flag- This flag indicates whether the trip was held in the memory before the trip started or not.

Trip_duration- duration of the trip in seconds.

## **EDA:**
Exploring the dataset we notice that we have some DateTime columns, we use only the pickup_datetime and obtain day, month, week, hour, and minute from the pickup_datetime. Furthermore, we add a distance_in_km from the pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude. 

<img width="581" alt="image" src="https://user-images.githubusercontent.com/71693871/206904523-b9b42e73-3db8-4b69-835e-9d39add64836.png">

On visualizing the distribution plot for the month and hour columns, we see that we have data for the first six months, and we also notice that the number of rides increases every six minutes. Also, the number of rides is maximum between 6 p.m. and 11 p.m, and the minimum number of rides is observed early in the morning.
We also notice that trip_duration is heavily rightly skewed but a simple log transformation makes it a normalized distribution.
We perform a regplot on the variables vs trip_duration and we notice that none of the variables varies linearly with the trip_duration. 

<img width="444" alt="image" src="https://user-images.githubusercontent.com/71693871/206904569-52763bc1-cdc8-45a7-98f6-0457521e18d3.png">

So, using linear regression may not fetch us the best results (however we will verify it). We further decide to apply a Decision tree, Random Forest model, and XGBoost model.
Further we added new columns named ‘is_holiday” which is binary containing information regarding whether the particular observation appeared on a holiday or not, but this column was highly unbalanced so we remove it.
Next column we added is distance, which is basically shortest angular distance between the pickup and dropoff location. This variable is not appropriate as it does not give the street distance. But Google maps API is chargeable, so we used havesine.


## **Null Values, outlier treatment and feature engineering:**

A good part of our problem is that the data is already cleaned for this project. So, we don’t have any null values or duplicated values. This can be further verified using df.isnull( ).sum( ) and df.duplicated( ). However, the number of passenger_count column contains values that are 0. Now, we have two options to fix it, one is to drop all such observations or replace them with the mode. We decide to replace the 0 values with 1, as the number of passengers logically shouldn't have any effect on trip duration. Unless and until it is large which will require a larger vehicle, such occurrences are very rare in our data
The passenger_count has values between 1 and 8. We assign value 1 to all the observations having passenger_count 1, and 2 to observations having passenger_count 2 to 5, and any observation having passenger_count more than 5 is given the value 2.
Next, we visualize the boxplot of trip_duration and realize that it has a lot of outliers and some are illogical, some trip durations (given in seconds) are well above 40 days. So we cap trip durations between 100 and 10000 seconds.
On performing a simple value count on ‘store_and_fwd_flag’ we note that only 0.005% of values are Y. So, we decide to drop this column. Further, we drop the columns pickup_datetime and dropoff_datetime.
 
## Linear Regression: 

As established linear regression is not a very ideal model for our dataset. However, we try to implement a linear regression. For this, we use the following from scikit library.

MinMaxScaler

train_test_split

LinearRegression

r2_score

mean_squared_error

In order to apply linear regression, we need to make sure the data doesn't suffer from multicollinearity. For this, we use VIF and remove those columns which show high and similar values.
At the end these variables remain in our dataset. 

<img width="319" alt="image" src="https://user-images.githubusercontent.com/71693871/206904680-a7f3cdfd-2c0d-4f25-92ce-84946b387d2a.png">

The target variable trip_duration is extremely right skewed. So we apply log transformation and it works perfectly.

As predicted linear regression yields very bad results.
The coefficient of determination or R2 value comes out to be 0.014, whereas the adjusted R2 on test set was -528.
So, we move on to lasso regression and the R2 and adjusted R2 obtained on test set are 0.014.
Next, we move on to ridge regression where the R2 and adjusted R2 values on test set obtained are 0.014.
At the end we apply elasticnet regression to get the adjusted R2 value on test set highly negative.
So, all in all liner regression did not work quite good. However it was fast and major problems were faced in removing correlation from the dataset. During which we also lost information regarding the pickup and dropoff location which may have played a pivotal role in determination of trip duration.

## **Decision Tree**
For the tree algorithms we did not use the same test and train dataset we used for linear regression. We use the dataset we had before we started removing multicollinearity from the dataset.
We didnot use GridSearchCV for hyperparameter tuning in the decision tree as it was taking too much time and we can use ensemble trees for better results. So we manually chose three pairs for max_depth, min_sample_split and min_sample_leaf and we go ahead with the best results. Since, our dataset is very large we keep the min_sample_split very large at 10000. This will also help our case as regression in decision tree takes averages such that the variance from the average for every point is minimum. So, for various vertices wich contains observations having a large range of trip_duration we can average them for 1000 observations.
The result obtained are as follows:

Train errors :

R2=0.6861247461299104

adjusted R2=0.686113389474079 

MSE=367.68784297220475

MAPE=0.31278393614284844


Test trrors :

R2=0.6850426044820102

adjusted R2=0.6850312086720558 

MSE=367.8704627667884

MAPE=0.3142893485875397

## **Random Forest Implementation**

We don’t make any changes to the dataset we used for Decision Tree implementation. This time we apply GridSearchCV for hyperparameter tuning, for such a large dataset it takes a lot of time around 12 hours.

Next, we consider the following hyperparameters:

n_estimators = 100

max_depth = 10

min_sample_split = 800

The results from Random Forest are:

Train errors :

R2=0.7129208679618806

adjusted R2=0.7129104808451558 

MSE=351.5865999320047

MAPE=0.299668472361449

Test errors :

R2=0.7110141538770642

adjusted R2=0.7110036977714805 

MSE=352.2726570146112

MAPE=0.301694963755809

## **XGBoost Implementation**

For XGBoost we try to implement bayesian optimization for hyperparameter tuning but it was taking a lot of time and frequent disconnections from collab was also notices. So we experimented with a few values of hyperparameters and settled on the following:

max_depth=6

n_estimators=100

learning_rate=0.1

The results from Random Forest are:

Train errors :

R2=0.7551077306162512

adjusted R2=0.7550988699074669 

MSE=324.7792467508347

MAPE=0.2725390847289322

Test trrors :

R2=0.7545482315087211

adjusted R2=0.7545393505561018 

MSE=324.7521957753851

MAPE=0.2737855267112696

## **Improvements:**

Applying bayesian optimization on the XGBoost model to improve the accuracy.
Having the variables temperature, snow level, whether the ride was an emergency or not and whether the cab had an accident during the ride or not.

## **Conclusion:**

After applying four models we can conclude that XGBoost worked the best, but it may be improved by Bayesian optimization. Using XGBoost we were able to account for 75% of the target variable using the independent variables and there was an average difference of 18 seconds in our prediction and actual value.


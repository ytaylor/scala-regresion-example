# scala-regresion-example
 The objective of this mini-project is to build (train) a linear regression model to predict the number of daily rentals (or schedules) in Washington's bicycle loan system (https://www.capitalbikeshare.com/)
 
The data set has been taken from: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

## Exploring the data
In this first stage, a univariate exploration is carried out to get an idea of the distribution of values of each variable and if there is any atypical. We have opted for the data set day, which has 731 instances, where each instance counts with 16 attributes, of them 9 are categorical and the rest are numeric. 


## Characterization of the variables
First we identify the predictor variables and the objective variables, ie that which has the output of what we want to predict, in turn we identify the type of data and the category of variables.

Types of variables:
* Predictor variables: season, holiday, weekend, workingday, weathersit, temp, atemp, hum, windspeed
* Output variables: casual, registered, cnt

Category of the variables:
* Qualitative variables: yr, season, hr, holiday, weekday, working day,
weathersit, instants and date.
* Numerical variables: temp, atemp, hum, windspeed, casual, registered, cnt

Observations have been made to the data to verify the quality of the data, which there are no missing values and the range of values of each variable. 

Range of variables
* temperature (between 0.02 and 1)
* temperature (between 0 and 1)
* humidity (between 0 and 1)
* winspedd (between 0.0000 and 0.8507)
* casual (between 2 and 3410)
* registered (between 20 and 6946)
* cnt (between 22 and 8714)

It can be seen that the data seems valid. Keep in mind that temp and temp have been divided by 41 (41 being the maximum registered temperature) and 50 (max. temperature) respectively. Also, the humidity has been divided by 100
therefore, any reading on the value of 1 would have been unusual. The last three ranges refer to the number of bicycles that have been used at any time during the two-year period.

## Preparation of features
As the data set there are categorical characteristics it is necessary to code them. For coding we have made a tobinary method that allows us to make a transformation where each qualitative variable with n categories makes n-1 appear
variables in the format (0,1,0) (0,0,0) ..., as necessary. 
With this method we have built a data structure composed of the label and features that will be later used for the construction of the model. 

## Model selection
For the selection of the model we use the p-values strategy. The p-value for each
term checks the null hypothesis that the coefficient is equal to zero (has no
effect). A low p-value less than 0.05 indicates that you can reject the null hypothesis.
In other words, a predictor that has a low p-value is likely to have a
significant addition to your model because changes in the value of the predictor are
related to changes in the response variable. Reciprocally, a p-value
large (negligible) suggests that changes in the predictor are not associated with
changes in the response.

The strategy consisted in making the model observing the answers obtained in
each iteration and go eliminating the variables or interactions that did not have a contribution
Finally, we have obtained a model with a r-squared coefficient of
0.8720.

It has been observed that, when obtaining the attributes, it was generally eliminated
those that are not highly correlated, for example, temperature and
atemp (normalized feeling temperature) are related so at the end
it is evident that it is not necessary both. That is why we have finally been
for the construction of the model with the following variables, and its extensive variables
obtained from the transformation of the qualitative characteristics:
* Season (independent variable)
* Year (independent variable)
* Month (independent variable)
* Weekday (independent variable)
* Workingday (independent variable)
* Weathersit (variable independiente)
* Temperature (variable independiente)
* Humidity (variable independiente)
* Windspeed (variable independiente)
* Season * temperatura (interacción)


## Validation of model hypotheses
To analyze the results of the regression, we must verify that the
hypothesis, that is to say that the normality, the linearity and the homogeneity of the variance.
For this we work with the residual which is the difference between the predicted values and
the actual values that were registered for the dependent variable.


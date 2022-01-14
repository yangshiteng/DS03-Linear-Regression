# 1. Linear Regression

## 1.1 Introduction

In statistics, linear regression is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression.[1] This term is distinct from multivariate linear regression, where multiple correlated dependent variables are predicted, rather than a single scalar variable.

Linear regression has many practical uses. Most applications fall into one of the following two broad categories:

- If the goal is prediction, linear regression can be used to fit a predictive model to an observed data set of values of the response and explanatory variables. After developing such a model, if additional values of the explanatory variables are collected without an accompanying response value, the fitted model can be used to make a prediction of the response.
- If the goal is to explain variation in the response variable that can be attributed to variation in the explanatory variables, linear regression analysis can be applied to quantify the strength of the relationship between the response and the explanatory variables, and in particular to determine whether some explanatory variables may have no linear relationship with the response at all, or to identify which subsets of explanatory variables may contain redundant information about the response.

Linear regression models are often fitted using the least squares approach, but they may also be fitted in other ways, such as by minimizing the "lack of fit" in some other norm (as with least absolute deviations regression), or by minimizing a penalized version of the least squares cost function as in ridge regression (L2-norm penalty) and lasso (L1-norm penalty). Conversely, the least squares approach can be used to fit models that are not linear models. Thus, although the terms "least squares" and "linear model" are closely linked, they are not synonymous.


## 1.2 Formulation

![image](https://user-images.githubusercontent.com/60442877/147891201-066a731d-6e34-4fdc-abce-7ab0e6420572.png)


## 1.3 Assumption

Standard linear regression models with standard estimation techniques make a number of assumptions about the predictor variables, the response variables and their relationship. Numerous extensions have been developed that allow each of these assumptions to be relaxed (i.e. reduced to a weaker form), and in some cases eliminated entirely. Generally these extensions make the estimation procedure more complex and time-consuming, and may also require more data in order to produce an equally precise model.

The following are the major assumptions made by standard linear regression models with standard estimation techniques (e.g. ordinary least squares):

- Linearity. This means that the mean of the response variable is a linear combination of the parameters (regression coefficients) and the predictor variables.
- Constant variance (a.k.a. homoscedasticity). This means that the variance of the errors does not depend on the values of the predictor variables. Thus the variability of the responses for given fixed values of the predictors is the same regardless of how large or small the responses are.  In order to check this assumption, a plot of residuals versus predicted values (or the values of each individual predictor) can be examined for a "fanning effect" (i.e., increasing or decreasing vertical spread as one moves left to right on the plot). A plot of the absolute or squared residuals versus the predicted values (or each predictor) can also be examined for a trend or curvature.
- Independence of errors. This assumes that the errors of the response variables are uncorrelated with each other. 
- Lack of perfect multicollinearity in the predictors. For standard least squares estimation methods, the design matrix X must have full column rank p; otherwise perfect multicollinearity exists in the predictor variables, meaning a linear relationship exists between two or more predictor variables. This can be caused by accidentally duplicating a variable in the data, using a linear transformation of a variable along with the original (e.g., the same temperature measurements expressed in Fahrenheit and Celsius), or including a linear combination of multiple variables in the model, such as their mean.

##  1.4 Regularization

There are extensions of the training of the linear model called regularization methods. These seek to both minimize the sum of the squared error of the model on the training data (using ordinary least squares) but also to reduce the complexity of the model (like the number or absolute size of the sum of all coefficients in the model).

Two popular examples of regularization procedures for linear regression are:

- Lasso Regression: where Ordinary Least Squares is modified to also minimize the absolute sum of the coefficients (called L1 regularization).
- Ridge Regression: where Ordinary Least Squares is modified to also minimize the squared absolute sum of the coefficients (called L2 regularization).

These methods are effective to use when there is collinearity in your input values and ordinary least squares would overfit the training data.

## 1.5 Co-efficient from Normal equations

![image](https://user-images.githubusercontent.com/60442877/147891701-e26a6652-50c1-4a50-b2cf-5b6071a39790.png)

## 1.6 Metrics for model evaluation

- R-Squared value： This value ranges from 0 to 1. Value ‘1’ indicates predictor perfectly accounts for all the variation in Y. Value ‘0’ indicates that predictor ‘x’ accounts for no variation in ‘y’.
- Regression sum of squares (SSR)： This gives information about how far estimated regression line is from the horizontal ‘no relationship’ line (average of actual output).
- Sum of Squared error (SSE)： How much the target value varies around the regression line (predicted value).
- Total sum of squares (SSTO)： This tells how much the data point move around the mean.
- Correlation co-efficient (r)： This is related to value of ‘r-squared’ which can be observed from the notation itself. It ranges from -1 to 1. r = (+/-) sqrt(r²). If the value of b1 is negative, then ‘r’ is negative whereas if the value of ‘b1’ is positive then, ‘r’ is positive. It is unitless.

## 1.7 Is the range of R-Square always between 0 to 1?

Value of R2 may end up being negative if the regression line is made to pass through a point forcefully. This will lead to forcefully making regression line to pass through the origin (no intercept) giving an error higher than the error produced by the horizontal line. This will happen if the data is far away from the origin.


# 2. Ridge and Lasso Regression: L2 and L1 Regularizationi

As I’m using the term linear, first let’s clarify that linear models are one of the simplest way to predict output using a linear function of input features.

![image](https://user-images.githubusercontent.com/60442877/147895633-58adeea5-fc2b-483c-abbe-bdb7563ad815.png)

In the equation above, we have shown the linear model based on the n number of features. Considering only a single feature as you probably already have understood that w[0] will be slope and b will represent intercept. Linear regression looks for optimizing w and b such that it minimizes the cost function. The cost function can be written as

![image](https://user-images.githubusercontent.com/60442877/147895651-e10579c2-0831-4ace-900f-51163a0bc675.png)

n the equation above I have assumed the data-set has M instances and p features. Once we use linear regression on a data-set divided in to training and test set, calculating the scores on training and test set can give us a rough idea about whether the model is suffering from over-fitting or under-fitting. The chosen linear model can be just right also, if you’re lucky enough! If we have very few features on a data-set and the score is poor for both training and test set then it’s a problem of under-fitting. On the other hand if we have large number of features and test score is relatively poor than the training score then it’s the problem of over-generalization or over-fitting. Ridge and Lasso regression are some of the simple techniques to reduce model complexity and prevent over-fitting which may result from simple linear regression.

## 2.1 Ridge Regression

In ridge regression, the cost function is altered by adding a penalty equivalent to square of the magnitude of the coefficients.

![image](https://user-images.githubusercontent.com/60442877/147895682-dbc3ac8d-7b12-435b-91b7-b519a1459e9c.png)

This is equivalent to saying minimizing the cost function under the condition as below

![image](https://user-images.githubusercontent.com/60442877/147895706-4b09c634-8abd-45df-9c57-25ad22d5584c.png)

So ridge regression puts constraint on the coefficients (w). The penalty term (lambda) regularizes the coefficients such that if the coefficients take large values the optimization function is penalized. So, ridge regression shrinks the coefficients and it helps to reduce the model complexity and multi-collinearity. When λ → 0 , the cost function becomes similar to the linear regression cost function. So lower the constraint (low λ) on the features, the model will resemble linear regression model. 

## 2.2 Lasso Regression 

The cost function for Lasso (least absolute shrinkage and selection operator) regression can be written as

![image](https://user-images.githubusercontent.com/60442877/147895775-1f9b460e-1474-46c5-9d29-49ebb66a067a.png)

Just like Ridge regression cost function, for lambda =0, the equation above reduces to the cost function of ordinary linear regression. The only difference is instead of taking the square of the coefficients, magnitudes are taken into account. This type of regularization (L1) can lead to zero coefficients i.e. some of the features are completely neglected for the evaluation of output. So Lasso regression not only helps in reducing over-fitting but it can help us in feature selection. 

## 2.3 Lasso regression can lead to feature selection whereas Ridge can only shrink coefficients close to zero.



# 3. Logistic regression

## 3.1 Introduction

In statistics, the logistic model (or logit model) is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc. Each object being detected in the image would be assigned a probability between 0 and 1, with a sum of one.

- Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression). Mathematically, a binary logistic model has a dependent variable with two possible values, such as pass/fail which is represented by an indicator variable, where the two values are labeled "0" and "1". In the logistic model, the log-odds (the logarithm of the odds) for the value labeled "1" is a linear combination of one or more independent variables ("predictors")
- The independent variables can each be a binary variable (two classes, coded by an indicator variable) or a continuous variable (any real value). The corresponding probability of the value labeled "1" can vary between 0 (certainly the value "0") and 1 (certainly the value "1"), hence the labeling; the function that converts log-odds to probability is the logistic function, hence the name. The unit of measurement for the log-odds scale is called a logit, from logistic unit, hence the alternative names. Analogous models with a different sigmoid function instead of the logistic function can also be used, such as the probit model; 


## 3.2 What is Wrong with Linear Regression for Classification?

The linear regression model can work well for regression, but fails for classification. Why is that? In case of two classes, you could label one of the classes with 0 and the other with 1 and use linear regression. Technically it works and most linear model programs will spit out weights for you. But there are a few problems with this approach:

A linear model does not output probabilities, but it treats the classes as numbers (0 and 1) and fits the best hyperplane (for a single feature, it is a line) that minimizes the distances between the points and the hyperplane. So it simply interpolates between the points, and you cannot interpret it as probabilities.

## 3.3 Types of Logistic Regression

1. Binary Logistic Regression
The categorical response has only two 2 possible outcomes. Example: Spam or Not

2. Multinomial Logistic Regression
Three or more categories without ordering. Example: Predicting which food is preferred more (Veg, Non-Veg, Vegan)

3. Ordinal Logistic Regression
Three or more categories with ordering. Example: Movie rating from 1 to 5

## 3.4 Decision Boundary

To predict which class a data belongs, a threshold can be set. Based upon this threshold, the obtained estimated probability is classified into classes.

Say, if predicted_value ≥ 0.5, then classify email as spam else as not spam.

Decision boundary can be linear or non-linear. Polynomial order can be increased to get complex decision boundary.




## 3.5 Logistic Model

![image](https://user-images.githubusercontent.com/60442877/147892500-285cc847-ada8-42d1-a96e-373164a8e026.png)


## 3.6 Model fitting

![image](https://user-images.githubusercontent.com/60442877/147892787-9940b042-9ea1-47c5-8f9e-41fed75f5ccb.png)


## 3.7 Multinomial Logistic Regression

Multinomial logistic regression is an extension of logistic regression that adds native support for multi-class classification problems.

Logistic regression, by default, is limited to two-class classification problems. Some extensions like one-vs-rest can allow logistic regression to be used for multi-class classification problems, although they require that the classification problem first be transformed into multiple binary classification problems.

Instead, the multinomial logistic regression algorithm is an extension to the logistic regression model that involves changing the loss function to cross-entropy loss and predict probability distribution to a multinomial probability distribution to natively support multi-class classification problems.

By default, logistic regression cannot be used for classification tasks that have more than two class labels, so-called multi-class classification.

Instead, it requires modification to support multi-class classification problems.

One popular approach for adapting logistic regression to multi-class classification problems is to split the multi-class classification problem into multiple binary classification problems and fit a standard logistic regression model on each subproblem. Techniques of this type include one-vs-rest and one-vs-one wrapper models.

An alternate approach involves changing the logistic regression model to support the prediction of multiple class labels directly. Specifically, to predict the probability that an input example belongs to each known class label.

The probability distribution that defines multi-class probabilities is called a multinomial probability distribution. A logistic regression model that is adapted to learn and predict a multinomial probability distribution is referred to as Multinomial Logistic Regression. Similarly, we might refer to default or standard logistic regression as Binomial Logistic Regression.

- Binomial Logistic Regression: Standard logistic regression that predicts a binomial probability (i.e. for two classes) for each input example.
- Multinomial Logistic Regression: Modified version of logistic regression that predicts a multinomial probability (i.e. more than two classes) for each input example.

Changing logistic regression from binomial to multinomial probability requires a change to the loss function used to train the model (e.g. log loss to cross-entropy loss), and a change to the output from a single probability value to one probability for each class label.

## 3.8 Video Tutorial 

###  3.81 https://www.youtube.com/watch?v=yIYKR4sgzI8

- Logistic Regression predicts whether something is True or False instead of predicting something continuous
- Instead of fitting a line to the data like linear regression, logistic regression fits an 'S' shaped 'logistic function'
![image](https://user-images.githubusercontent.com/60442877/149512667-7a4917f2-d9ae-4c5d-802e-f73ad85a73c0.png)
- Logistic regression's ability to provide probabilities and classify new samples using continuous and discrete measurements make it a popular machine learning method
- One big difference between linear regression and logistic regression is how the line is fit to the data
- With linear regression, we fit the line using 'Least Squares', we also use the residuals to calculate the R-square to compare simple models to complicated models
- However, logistic regression doesn't have the same concept of a 'residual', so it can't use least squares and it can't calculate R-square, instead, it uses 'Maximum Likelihood'

### 3.82 https://www.youtube.com/watch?v=vN5cNN2-HWE

- Logistic Regression is a specific type of Generalized Linear Model
![image](https://user-images.githubusercontent.com/60442877/149521205-058cdb64-cc53-4842-ae2b-c0a40749cce0.png)
- In terms of the coefficients, logistic regression is the exact same as good linear regresssion models except the coefficients are in terms of the log(odds)

### 3.83 https://www.youtube.com/watch?v=BfKanl1aSG0



# 4. Generalized Linear Model

In statistics, a generalized linear model (GLM) is a flexible generalization of ordinary linear regression. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

## 4.1 Overview

![image](https://user-images.githubusercontent.com/60442877/147896503-1c9ead21-8ee7-4536-989a-2183e172b75e.png)

## 4.2 Model components

![image](https://user-images.githubusercontent.com/60442877/147896518-d703ce12-c883-4cf6-8d1d-2a731ef8443d.png)

## 4.3 Common distributions with typical uses and canonical link functions

![image](https://user-images.githubusercontent.com/60442877/147896623-a02f68af-b8e2-4e59-b4d0-e1db0e0fa18d.png)






# Machine Learning Classification with Logistic Regression

## Overview

Classification is a machine learning method used to predict discrete values (categories or labels). The goal is to determine which category or class a new data point belongs to, using a binary or multi-class classification approach. Classification helps in various domains such as detecting market declines, identifying suspicious activities, and preventing banking fraud.

In this project, we use **Logistic Regression** to classify data into two categories, where the target variable is categorical.

## Table of Contents

- [Classification](#classification)
- [Logistic Regression](#logistic-regression)
- [Activities and Model Implementation](#activities-and-model-implementation)
- [Scaling and Normalization](#scaling-and-normalization)
- [Linear Classifiers and SVM](#linear-classifiers-and-svm)
- [Support Vector Machines (SVM)](#support-vector-machines-svm)
- [Accuracy and Model Evaluation](#accuracy-and-model-evaluation)
- [Random Forest and KNN](#random-forest-and-knn)
- [Cross-Validation and VIF](#cross-validation-and-vif)

## Classification

Classification is a method used to predict discrete variables, where the values cannot be divided or averaged. For instance, in a binary classification, the model may predict whether an event will happen ("yes" or "no"), like whether a person owns a car.

### Examples of Classification Problems:
- Whether someone will purchase a product ("yes" or "no").
- Whether a loan will be approved ("approved" or "denied").
- Whether an email is spam or not.

### Logistic Regression

Logistic Regression is a statistical method used for binary classification. It predicts the probability of an outcome based on input data and is widely used in many fields, especially for classifying dichotomous outcomes (like fraud detection or medical diagnosis).

## Activities and Model Implementation

### Activity 1

- **Target Variable**: The column we are trying to predict is the "Category" column.
- **Test-Train Split**: We split the data into training (75%) and testing (25%) sets.

### Activity 2

1. **Value Counts**: Count the occurrences of each category in the target column.
2. **Train-Test Split**: Perform a train-test split of the dataset.
3. **Logistic Regression Model**: Fit a Logistic Regression model using the training data.

#### Logistic Regression Model:
```python
from sklearn.linear_model import LogisticRegression

# Logistic Regression model
log_reg = LogisticRegression(max_iter=100)
log_reg.fit(X_train, y_train)
Max Iter: By default, the maximum number of iterations is set to 100, but you can adjust it as needed.
Activity 3: Scaling
Scaling helps adjust the data so that all features have a similar numeric scale. This is important because some algorithms can be sensitive to the scale of input features.

You can use Min-Max Scaling or Standard Scaling, depending on your preference.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Activity 4
Try the Logistic Regression model without scaling:

no_scale_model = LogisticRegression(max_iter=128)
no_scale_model.fit(X_train, y_train)
print(no_scale_model.score(X_test, y_test))
Score: A score of 0.89 is achieved, which is very similar to the model with scaling.
Linear Classifiers and SVM

Linear classifiers aim to find a hyperplane (line) that separates two groups of data. Support Vector Machines (SVM) are used for multi-dimensional data, and they find a hyperplane in a higher-dimensional space. SVM models can be either linear or non-linear, depending on the kernel used.

SVM (Support Vector Machines): SVMs create hyperplanes to classify data points. They are effective in high-dimensional spaces.
Support Vector Machines (SVM)

SVM is a supervised learning model that can be used for both classification and regression. It creates a decision boundary (hyperplane) to separate different classes in the data.

Model Fit-Predict Pattern in Sklearn
Scikit-learn follows a pattern of model-fit-predict, which helps train, test, and evaluate machine learning models.

Accuracy and Model Evaluation
The accuracy of a classification model is the proportion of correct predictions. It can be calculated using the following formula:

Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
Accuracy is a commonly used metric to evaluate classification models.

Random Forest and KNN

Random Forest
Random Forest is one of the most widely used models in industry. It is an ensemble method that combines multiple decision trees to improve accuracy.

KNN (K-Nearest Neighbors)
KNN is a simple and intuitive algorithm used for classification and regression. It predicts the class of a data point based on the majority class of its nearest neighbors.

Cross-Validation and VIF

Cross-Validation
Cross-validation is a technique used to evaluate how well a machine learning model generalizes to an independent dataset. It helps avoid overfitting and underfitting by splitting the data into multiple training and testing sets.

Variance Inflation Factor (VIF)
The Variance Inflation Factor (VIF) measures how much the variance of a regression coefficient is inflated due to collinearity with other predictors. It is used to detect multicollinearity in a regression model.

Formula for Adjusted R²:

Adjusted R² = 1 - [(n - 1) / (n - p - 1)] * (1 - R²)
Where:
n = Number of data points
p = Number of features
R² = R-squared value
VIF Formula:
VIF = 1 / (1 - R²)
A high VIF value indicates multicollinearity, which can distort the interpretation of regression coefficients.

Conclusion

Classification models are powerful for making predictions about discrete categories.
Logistic Regression is an effective tool for binary classification.
Random Forest and SVM are also popular methods in classification tasks.
Scaling ensures the model is not biased by the magnitude of features.
Cross-validation and VIF are key tools to ensure robust model performance.
References

Scikit-learn documentation: https://scikit-learn.org/
"Introduction to Machine Learning with Python" by Andreas C. Müller & Sarah Guido.

### Explanation of Markdown Syntax:
- **Headers**: Use `#` for main headers, `##` for sub-headers, and so on.
- **Bold Text**: Use `**` around words you want to make bold.
- **Code Blocks**: Use triple backticks (```) before and after code blocks for code formatting.
- **Links**: Use `[text](URL)` for hyperlinks.

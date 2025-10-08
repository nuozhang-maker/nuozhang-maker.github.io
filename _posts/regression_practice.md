---
layout: post
author: Nuo Zhang
tags: [regression, practice]
---

# import library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# Create DataFrame
data = pd.read_csv('train.csv')
# Display the DataFrame
data.head()
# creating a backup copy of the data 
data_original = data.copy()
# Populating null Age values with the average age by sex, Pclass, 
data['Age'] = data.groupby(['Sex', 'Pclass'],group_keys=False)['Ag
# Plot before and after imputation
fig, axes = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
sns.histplot(data_original['Age'], ax=axes[0], kde=True, color='re
sns.histplot(data['Age'], kde=True, ax=axes[1], color='green').set
plt.show()
PassengerId
Survived
Pclass
Name
Sex
Age
SibSp
Parch
0
1
0
3
Braund,
Mr. Owen
Harris
male
22.0
1
0
1
2
1
1
Cumings,
Mrs. John
Bradley
Florence
Briggs
Th...
female
38.0
1
0
2
3
1
3
Heikkinen,
Miss.
Laina
female
26.0
0
0
3
4
1
1
Futrelle,
Mrs.
Jacques
Heath
Lily May
Peel)
female
35.0
1
0
4
5
0
3
Allen, Mr.
William
Henry
male
35.0
0
0
In [2]:
In [3]:
In [4]:
Out[4]:
In [5]:
In [6]:
In [7]:

Explanation of Not Using The "Survived" Field
Explanation:
Using the Survived field for imputing the Age field can lead to data leakage.
Data leakage occurs when information from outside the training dataset is
used to create the model. This can lead to overly optimistic performance
estimates and poor generalization to new data.
Comments:
Using the Survived field to impute Age can introduce bias because the

survival status might be influenced by the age of the passengers, thereby
distorting the model's understanding of the relationship between age and
survival.
Chart Questions
What does the plt.subplots function do?
This is a new function you haven't seen before, how would you find out
more about it?
Regression Model
Explanation:
A regression model, such as logistic regression, can be used to predict a
binary outcome (like survival).
# data = pd.get_dummies(data, columns=['Sex', 'Embarked'])
Step 1 Convert all categorical dimensions to numerical values.
Why?
Regression algorithms are based on mathematical operations and require
numerical input. Categorical variables, which represent qualitative data,
cannot be directly processed by these algorithms. Converting categorical
variables into numerical formats allows the model to interpret and analyze
relationships effectively.
# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_firs
Practical Question
What did the pd.get_dummies function do?
How would you find out?
#add your answer here
Step 2 Separate the data into features and target variables
Why
Separating the data into features (input variables) and target variables
(output variable) clearly defines what the model needs to predict. Features
In [8]:
In [9]:
In [10]:

provide the information used to make predictions, while the target variable
is the outcome the model aims to predict.
# Define features we will use for the model
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', '
# define the target variable 
y = data['Survived']
Step 3 Separate the feature and target dimensions into train and test
Why
Separating data into training and testing sets allows us to validate the
model's performance. Training the model on one subset and testing it on
another helps assess how well the model generalizes to new, unseen data.
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_siz
Step 4 Training the logistic regression model
# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
/usr/lib/python3.13/site-packages/sklearn/linear_model/_logistic.p
y:473: ConvergenceWarning: lbfgs failed to converge after 200 itera
tion(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Increase the number of iterations to improve the convergence (max_i
ter=200).
You might also want to scale the data as shown in:
   https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver optio
ns:
   https://scikit-learn.org/stable/modules/linear_model.html#logis
tic-regression
 n_iter_i = _check_optimize_result(
A little bit of theory
max_iter Parameter:
This parameter sets the maximum number of iterations that the optimization
algorithm can run to converge to the best solution. During each iteration, the
In [11]:
In [12]:
In [13]:
Out[13]:
▾LogisticRegression
?
i
▸ Parameters

algorithm updates the coefficients slightly, moving towards the direction that
reduces the error.
If the algorithm converges (i.e., the changes in the coefficients become very
small) before reaching the maximum number of iterations, it stops early. If
the algorithm does not converge within the specified number of iterations, it
stops and may not have found the best solution. This can happen if the data
is complex or the learning rate is not well-tuned.
Practical Question
Why are we using a logistic regression model in this situation?
add your answer here
Step 5 Use the trained model to predict the output
# Predict on the test set
y_pred = model.predict(X_test)
Step 6 Compare the results of the predicted output with the actual answers
i.e. y_pred v y_test
# Evaluate the model using accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
Accuracy: 0.8555555555555555
Confusion Matrix:
[[46  8]
[ 5 31]]
Technical Reminder
Confusion Matrix a tool typically used to evaluate the performance of
classification models, not regression models. It summarizes the number of
correct and incorrect predictions made by the model, comparing actual
target values with predicted values.
Accuracy Score a metric used to evaluate the correctness of predictions.
For classification, it is the ratio of the number of correct predictions to the
total number of predictions.
image: Research Gate 
Step 7 Calculate feature importance
In [14]:
In [15]:

Why
Calculating feature importance helps in understanding which features have
the most significant impact on the model's predictions. By identifying the
most important features, we can keep the most relevant features and
improve model performance.
# Calculate feature importance
feature_importance = model.coef_[0]
# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)
# Plot feature importance
plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()
Understanding Feature Importance Scores
Positive Importance Score: A positive coefficient indicates that as the
feature value increases, the likelihood of the positive class increases
(assuming binary logistic regression). In other words, higher values of this
feature are associated with a higher probability of the target being 1 (or the
positive class).
Negative Importance Score: A negative coefficient indicates that as the
In [16]:

feature value increases, the likelihood of the positive class decreases. This
means that higher values of this feature are associated with a higher
probability of the target being 0 (or the negative class).
Step 8 Transform the test data into the format required for the model
# Import new test data
test_data = pd.read_csv('test.csv')
# Populating null Age values with the average age by sex, Pclass, 
test_data['Age'] = test_data.groupby(['Sex', 'Pclass'],group_keys=
# check for null values 
test_data.isnull().sum()
PassengerId      0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
# using an average of sex and PClass for the missing fare value
test_data['Fare'] = test_data.groupby(['Sex', 'Pclass'],group_keys
# Preprocess the test data in the same way as the training data
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'],
# Ensure the test data has the same columns as the training data
test_data = test_data.reindex(columns=X.columns, fill_value=0)
# Predict on the new test data
test_predictions = model.predict(test_data)
# adding the survived field back to the test data
test_data['Survived_predicated'] = test_predictions
Practical Example
Create a new column FamilySize in the DataFrame, which is the sum of SibSp
and Parch.
Then, create a regression model to predict the Survived field using Pclass,
Age, FamilySize, Fare, Sex_male, Embarked_Q, and Embarked_S.
In [17]:
In [18]:
In [19]:
Out[19]:
In [20]:
In [21]:
In [22]:
In [23]:
In [24]:

# add your answer below
data['FamilySize'] = data.SibSp + data.Parch
data.head()
# Define features we will use for the model
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', '
# define the target variable 
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_siz
X_train.shape
(801, 8)
y_train.shape
(801,)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
PassengerId
Survived
Pclass
Name
Age
SibSp
Parch
Ticke
0
1
0
3
Braund,
Mr. Owen
Harris
22.0
1
0
A/5 2117
1
2
1
1
Cumings,
Mrs. John
Bradley
Florence
Briggs
Th...
38.0
1
0
PC 1759
2
3
1
3
Heikkinen,
Miss.
Laina
26.0
0
0
STON/O
310128
3
4
1
1
Futrelle,
Mrs.
Jacques
Heath
Lily May
Peel)
35.0
1
0
11380
4
5
0
3
Allen, Mr.
William
Henry
35.0
0
0
37345
In [25]:
Out[25]:
In [28]:
Out[28]:
In [29]:
Out[29]:
In [30]:

/usr/lib/python3.13/site-packages/sklearn/linear_model/_logistic.p
y:473: ConvergenceWarning: lbfgs failed to converge after 200 itera
tion(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Increase the number of iterations to improve the convergence (max_i
ter=200).
You might also want to scale the data as shown in:
   https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver optio
ns:
   https://scikit-learn.org/stable/modules/linear_model.html#logis
tic-regression
 n_iter_i = _check_optimize_result(
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
Accuracy: 0.8555555555555555
Confusion Matrix:
[[46  8]
[ 5 31]]
feature_importance = model.coef_[0]
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()
Out[30]:
▾LogisticRegression
?
i
▸ Parameters
In [31]:
In [32]:
In [33]:

test_data = pd.read_csv('test.csv')
test_data['Age'] = test_data.groupby(['Sex', 'Pclass'],group_keys=
In [0]:


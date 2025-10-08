---
layout: post
author: Nuo Zhang
tags: [airbnb, prediction]
---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import graphviz
# Load the airbnb data
airbnb_df = pd.read_csv('airbnb_data.csv')
# Load the location data
location_df = pd.read_csv('location.csv')
# merge the datasets 
df = pd.merge(airbnb_df,location_df,
                 left_on='id',right_on='id',how='inner')
df.head()
# Select features for prediction, now including neighbourhood
features = ['room_type', 'neighbourhood', 'minimum_nights', 'numbe
            'reviews_per_month', 'calculated_host_listings_count',
# Create a new dataframe with only the selected features and price
df_selected = df[features + ['price']]
# Drop rows with NaN values
id
name
host_id
host_name
room_type
price
minimum_nigh
0
11156
An Oasis
in the City
40855
Colleen
Private
room
65.0
1
14250
Manly
Harbour
House
55948
Heidi
Entire
home/apt
600.0
2
15253
Unique
Designer
Rooftop
Apartment
in City
Loca...
59850
Morag
Private
room
118.0
3
58506
Studio
Yindi @
Mosman,
Sydney
279955
John
Entire
home/apt
190.0
4
68999
A little bit
of Sydney
- Australia
333581
Bryan
Private
room
115.0
In [5]:
In [6]:
Out[6]:
In [7]:
df_clean = df_selected.dropna()
# Convert categorical variables to dummy variables
df_encoded = pd.get_dummies(df_clean, columns=['room_type', 'neigh
# Separate features (X) and target variable (y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_siz
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
# Evaluation function using Mean Absolute Percentage Error (MAPE)
def evaluate_model(y_true, y_pred, model_name):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{model_name} - Mean Absolute Percentage Error: {mape:.
evaluate_model(y_test, lr_predictions, "Linear Regression")
evaluate_model(y_test, rf_predictions, "Random Forest")
Linear Regression - Mean Absolute Percentage Error: 77.52%
Random Forest - Mean Absolute Percentage Error: 60.82%
# Visualize predictions vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, lr_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()
In [8]:
In [9]:
In [10]:
# Visualize predictions vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()
plt.show()
# Exporting the tree from the Random Forest model
tree = rf_model.estimators_[99]
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=X.columns, 
In [11]:
In [13]:
                           filled=True, rounded=True, 
                           special_characters=True,
                           max_depth=4)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="pdf")
# Displaying the decision tree
print('decision tree saved')
-------------------------------------------------------------------
--------
NameError                                 Traceback (most recent ca
ll last)
Cell In[13], line 3
     1 # Exporting the tree from the Random Forest model
     2 tree = rf_model.estimators_[99]
----> 3 dot_data = export_graphviz(tree, out_file=None, 
     4                            feature_names=X.columns, 
     5                            filled=True, rounded=True, 
     6                            special_characters=True,
     7                            max_depth=4)
     8 graph = graphviz.Source(dot_data)
     9 graph.render("decision_tree", format="pdf")
NameError: name 'export_graphviz' is not defined

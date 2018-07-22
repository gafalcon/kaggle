# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:39:21 2018

@author: gafalcon
"""
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
y = df.Cover_Type
df.drop(["Id", "Cover_Type"], axis=1, inplace=True)
test = pd.read_csv("test.csv")
test_Ids = test.Id
test.drop(["Id"], axis=1, inplace=True)
def add_features(df):
    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
    df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    df['ele_vert'] = df.Elevation-df.Vertical_Distance_To_Hydrology

    df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
    df.slope_hyd=df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    #Mean distance to Amenities 
    df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 
    #Mean Distance to Fire and Water 
    df['Mean_Fire_Hyd']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2
    return df

df = add_features(df)
df.to_csv("train_extra_features.csv", index=False)


def rmsle_cv(model, x, y, n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def perform_grid_search(clf, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(clf, param_grid, cv=cv)
    grid_search.fit(X, y)
    print ("Best params", grid_search.best_params_)
    return grid_search.best_estimator_

def cv_score(clf, X, y, n_splits=5, scoring=None):
    # cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=n_splits, scoring=scoring)
    #print ("Scores with C=",C, scores)
    print("Scores: ", scores)
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

def predict(clf, X_test, csv_name):
    global test_Ids
    predictions = clf.predict(X_test)
    pred_df = pd.DataFrame()
    pred_df["Id"] = test_Ids
    pred_df["Cover_Type"] = predictions
    pred_df.to_csv(csv_name, index=False)

def train_valid_score(model, X, y, test_size=0.3):
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size)
    model.fit(x_train, y_train)
    print ("Train: ", model.score(x_train, y_train), "Valid:", model.score(x_valid, y_valid))

classifiers = [
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier(),
    LGBMClassifier()
    ]


for classifier in classifiers:
    print (classifier)
    cv_score(classifier, df, y, scoring="accuracy")

rf_param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [80, 100,150], 'min_samples_leaf': [3,5,7], 'max_features':[0.5, 0.7, 0.3]},
    # then try 6 (2×3) combinations with bootstrap set as False
    #{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

rf_clf_n = perform_grid_search(RandomForestClassifier(), rf_param_grid, df, y)
cv_score(rf_clf_n, df, y)


rf_clf = RandomForestClassifier(max_features=0.3, min_samples_leaf=3, n_estimators=100)
rf_clf.fit(df,y)
cv_score(rf_clf, df, y) #0l79 +- 0.07
train_valid_score(RandomForestClassifier(max_features=0.3, n_estimators=100, min_samples_leaf=3), df.values, y.values, test_size=0.2)


import numpy as np
import pandas as pd
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

#Loading
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
# Split into features and labels
y = train.Survived

X_train = train.drop(["Survived"], axis=1)
X_test = test

dataset = pd.concat([X_train, X_test])


print('Columns with null values:\n', dataset.isnull().sum())

titleDict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Sir",
    "Don": "Sir",
    "Sir": "Sir",
    "Dr": "Dr",
    "Rev": "Rev",
    "theCountess": "Lady",
    "Dona": "Lady",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Lady"
}

def splitName(s):
    """
    Extract title from name, replace with value in title dictionary. Also 
    return surname.
    """
    s = s.replace('.', '')# Remove '.' from name string
    s = s.split(' ')# Split on spaces
    surname = s[0]# get surname
    title = [t for k,t in titleDict.items() if str(k) in s]
    if title == []:
        title = 'Other'
    else:
        title = title[0]# Title is a list, so extract contents
    return surname.strip(','), title

def getAttrFromName(X):
    out = X["Name"].apply(splitName)
    out = out.apply(pd.Series)
    # out.columns = ['Surname', 'Title']
    X[["Surname", "Title"]] = out
    # X = X.drop(["Name"], axis=1)
    X = X.drop(["Name", "Surname"], axis=1)
    return X

def groupby(X, column):
    print (X[[column, 'Survived']].groupby([column], as_index=False).mean())

def getFamilyAttr(X):
    X["FamilySize"] = X["Parch"] +  X["SibSp"] + 1
    X["IsAlone"] = 0
    X.loc[X["FamilySize"] == 1, "IsAlone"] = 1
    X = X.drop(["SibSp", "Parch"], axis=1)
    return X

def fillNa(X):
    num_columns = ["Fare", "Parch", "SibSp"]
    X[num_columns] = X.loc[:, num_columns].fillna(X[num_columns].median())
    age_avg = X['Age'].mean()
    age_std = X['Age'].std()
    age_null_count = X['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # X.loc[:, 'Age'][np.isnan(X['Age'])] = age_null_random_list
    X.loc[np.isnan(X['Age']), 'Age'] = age_null_random_list
    X['Age'] = X['Age'].astype(int)
    cat_columns = ["Sex", "Pclass", "Embarked"]
    for col in cat_columns:
        X.loc[:, col].fillna(X[col].mode()[0], inplace=True)
    # X[cat_columns] = X.loc[:, cat_columns].fillna(X[cat_columns].mode()[0])
    return X


def drop_columns(X):
    return X.drop(["Cabin", "Ticket", "PassengerId"], axis=1)


def missing_data(train):
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


def categorize(X):
    X["Age_cat"] = pd.qcut(X["Age"], 5, labels=False, duplicates="drop")
    X["Fare_cat"] = pd.qcut(X["Fare"],5, labels=False, duplicates="drop")
    return X.drop(["Age", "Fare"], axis=1)


def pipeline_binned(X):
    global X_train
    X = drop_columns(X)
    X = fillNa(X)
    X = getAttrFromName(X)
    X = getFamilyAttr(X)
    X = categorize(X)
    X = pd.get_dummies(X, columns=["Sex", "Embarked", "Title", "Age_cat", "Fare_cat"], drop_first=True)
    columns_to_scale = ["FamilySize"]
    sc_X = StandardScaler()
    X[columns_to_scale] = sc_X.fit_transform(X[columns_to_scale])
    X_train = X.iloc[:len(X_train), :]
    X_test = X.iloc[len(X_train):, :]
    return X_train, X_test
    return X

def pipeline(X):
    global X_train
    X = drop_columns(X)
    X = fillNa(X)
    X = getAttrFromName(X)
    X = getFamilyAttr(X)
    print (X.columns)
    X = pd.get_dummies(X, columns=["Sex", "Embarked", "Title"], drop_first=True)
    print (X.columns)
    columns_to_scale = ["Age", "Fare", "FamilySize"]
    sc_X = StandardScaler()
    X[columns_to_scale] = sc_X.fit_transform(X[columns_to_scale])
    X_train = X.iloc[:len(X_train), :]
    X_test = X.iloc[len(X_train):, :]
    return X_train, X_test


df = dataset.copy()
df2 = dataset.copy()
X_train_n, X_test_n = pipeline(df)
X_train_binned, X_test_binned = pipeline_binned(df2)

# Prediction time
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [10, 30, 50, 70, 80, 90, 100], 'max_depth': [3, 5, 7, 10, 15]},
    # then try 6 (2×3) combinations with bootstrap set as False
    #{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

from sklearn.model_selection import cross_val_score, ShuffleSplit


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def perform_grid_search(clf, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(clf, param_grid, cv=cv)
    grid_search.fit(X, y)
    print ("Best params", grid_search.best_params_)
    return grid_search.best_estimator_

def cv_score(clf, X, y, n_splits=5):
    # cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=n_splits, scoring='f1_macro')
    #print ("Scores with C=",C, scores)
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

classifiers = [
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(),
    XGBClassifier()
    ]


for classifier in classifiers:
    print (classifier)
    cv_score(classifier, X_train_n, y)


# predictions = best_rf_estimator.predict(X_test)
xgb_clf = perform_grid_search(XGBClassifier(), param_grid, X_train_n, y)
cv_score(xgb_clf, X_train_n, y)

xgb_clf_bins = perform_grid_search(XGBClassifier(), param_grid, X_train_binned, y)
cv_score(xgb_clf_bins, X_train_binned, y)

rf_clf_n = perform_grid_search(RandomForestClassifier(), param_grid, X_train_n, y)
cv_score(rf_clf_n, X_train_n, y)

rf_clf_bin = perform_grid_search(RandomForestClassifier(), param_grid, X_train_binned, y)
cv_score(rf_clf_n, X_train_binned, y)

from sklearn.ensemble import VotingClassifier
estimators = [("xgb", xgb_clf),
              ("rf", rf_clf_n),
              ("ada", AdaBoostClassifier()),
              ("gb",GradientBoostingClassifier()),
              ("lr", LogisticRegression()),
              ("dt", DecisionTreeClassifier())]
voting_clf = VotingClassifier(estimators)
voting_clf.fit(X_train_n, y)
cv_score(voting_clf, X_train_n, y)

def predict(clf, X_test, csv_name):
    global test
    predictions = clf.predict(X_test)
    pred_df = pd.DataFrame()
    pred_df["PassengerId"] = test.PassengerId
    pred_df["Survived"] = predictions
    pred_df.to_csv(csv_name, index=False)

predict(voting_clf, X_test_n, "predv8_voting.csv")



## Stacking
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.5
X_train, X_test, Y_train, Y_test = train_test_split(X_train_n, y, test_size = TEST_SIZE)
estimators = [("xgb", XGBClassifier()),
              ("rf", RandomForestClassifier()),
              ("ada", AdaBoostClassifier()),
              ("gb",GradientBoostingClassifier()),
              ("lr", LogisticRegression()),
              ("dt", DecisionTreeClassifier()),
              ("svc", SVC(probability=True))
              ]

Stack_train = {}
Stack_test = {}
for estimator in estimators:
    estimator[1].fit(X_train, Y_train)
    Stack_train[estimator[0]] = estimator[1].predict(X_test)
    Stack_test[estimator[0]] = estimator[1].predict(X_test_n)
df_stack_train = pd.DataFrame(Stack_train)
df_stack_test = pd.DataFrame(Stack_test)
ensembled_clf = perform_grid_search(XGBClassifier(), param_grid, df_stack_train, Y_test, cv=15)
cv_score(ensembled_clf, df_stack_train, Y_test, n_splits=15)
predict(ensembled_clf, df_stack_test, "predv9_stacking.csv")

















train_pred = pd.DataFrame({"xgb_n": xgb_clf.predict(X_train_n),
                           "xgb_bin": xgb_clf_bins.predict(X_train_binned),
                           "rf": rf_clf_n.predict(X_train_n),
                           "rf_bins": rf_clf_bin.predict(X_train_binned)})

ensembled_clf = perform_grid_search(XGBClassifier(), param_grid, train_pred, y, cv=15)
cv_score(ensembled_clf, train_pred, y, n_splits=15)

y_pred = pd.DataFrame({
    "xgb_n": xgb_clf.predict(X_test_n),
    "xgb_bin": xgb_clf_bins.predict(X_test_binned),
    "rf": rf_clf_n.predict(X_test_n),
    "rf_bins": rf_clf_bin.predict(X_test_binned)
})

predict(ensembled_clf, y_pred, "predv8_voting.csv")

p_train = {}
p_test = {}
for estimator in estimators[2:]:
    estimator[1].fit(X_train_n, y)
    p_train[estimator[0]] = estimator[1].predict(X_train_n)
    p_test[estimator[0]] = estimator[1].predict(X_test_n)
p_train["xgb"] = xgb_clf.predict(X_train_n)
p_train["rf_clf"] = rf_clf_n.predict(X_train_n)
p_test["xgb"] = xgb_clf.predict(X_test_n)
p_test["rf_clf"] = rf_clf_n.predict(X_test_n)

df_p_train = pd.DataFrame(p_train)
df_p_test = pd.DataFrame(p_test)
ensembled_clf = perform_grid_search(XGBClassifier(), param_grid, df_p_train, y, cv=15)
cv_score(ensembled_clf, df_p_train, y, n_splits=15)
predict(ensembled_clf, df_p_test, "predv7_ensemble_xgb.csv")

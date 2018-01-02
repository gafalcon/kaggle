import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib
df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

# Drop id
df_train.drop("Id", axis=1, inplace=True)
df_test.drop("Id", axis=1, inplace=True)
# Drop columns with large num of missing values
columns_to_drop = ["Alley", "PoolQC", "Fence", "MiscFeature"]
df_train.drop(columns_to_drop, axis=1, inplace=True)
df_test.drop(columns_to_drop, axis=1, inplace=True)

def missing_data(train):
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data[missing_data["Total"] > 0]

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);


# TotalBsmtSF and 1stFlrSF are highly correlated. I'll delete one of these features
df_train.drop(["1stFlrSF"], axis=1, inplace=True)
df_test.drop(["1stFlrSF"], axis=1, inplace=True)

# GarageCars and GarageArea are highly corr. Delete one
df_train.drop(["GarageArea"], axis=1, inplace=True)
df_test.drop(["GarageArea"], axis=1, inplace=True)

# Delete GarageYrBlt highly correlated to YrBuilt
df_train.drop(["GarageYrBlt"], axis=1, inplace=True)
df_test.drop(["GarageYrBlt"], axis=1, inplace=True)

# Cant see significative correlation between FireplaceQu and Saleprice, Gonna Delete it
# sns.violinplot(x="FireplaceQu", y="SalePrice", data=df_train)

df_train.drop(["FireplaceQu"], axis=1, inplace=True)
df_test.drop(["FireplaceQu"], axis=1, inplace=True)

numeric = df_train.select_dtypes(include=[np.number])
category = df_train.select_dtypes(exclude=[np.number])


# Select columns with low correlation
cors = numeric.corr()["SalePrice"]
low_cor_columns = list(cors[abs(cors) < 0.2].index.values)
# sns.pairplot(numeric[low_cor_columns[4:8]+ ["SalePrice"]], dropna=True)

# Delete low correlated features
df_train.drop(low_cor_columns, axis=1, inplace=True)
df_test.drop(low_cor_columns, axis=1, inplace=True)

# There seem to be outliers in SalePrice, Ill delete them
outliers = df_train[df_train["SalePrice"] > 514000].index
df_train.drop(outliers, inplace=True)
outliers = df_train[df_train["LotFrontage"] > 250].index
df_train.drop(outliers, inplace=True)

# Fill LotFrontage nans
lf_avg = df_train['LotFrontage'].mean()
lf_std = df_train['LotFrontage'].std()
lf_null_count = df_train['LotFrontage'].isnull().sum()
lf_null_random_list = np.random.randint(lf_avg - lf_std, lf_avg + lf_std, size=lf_null_count)
# df_train.loc[:, 'Age'][np.isnan(df_train['Age'])] = age_null_random_list
df_train.loc[np.isnan(df_train['LotFrontage']), 'LotFrontage'] = lf_null_random_list

lf_null_count = df_test['LotFrontage'].isnull().sum()
lf_null_random_list = np.random.randint(lf_avg - lf_std, lf_avg + lf_std, size=lf_null_count)
# df_train.loc[:, 'Age'][np.isnan(df_train['Age'])] = age_null_random_list
df_test.loc[np.isnan(df_test['LotFrontage']), 'LotFrontage'] = lf_null_random_list
# Create garage? feature
df_test["GarageCars"] = df_test["GarageCars"].fillna(df_test.GarageCars.median())
df_train["Garage"] = 0
df_train.loc[df_train.GarageCars > 0, "Garage"] = 1
df_test["Garage"] = 0
df_test.loc[df_test.GarageCars > 0, "Garage"] = 1

df_train.loc[:, ["GarageCond", "GarageType", "GarageQual", "GarageFinish"]] = df_train.loc[:, ["GarageCond", "GarageType", "GarageQual", "GarageFinish"]].fillna("NoGarage")
df_test.loc[:, ["GarageCond", "GarageType", "GarageQual", "GarageFinish"]] = df_test.loc[:, ["GarageCond", "GarageType", "GarageQual", "GarageFinish"]].fillna("NoGarage")

# Delete bsmt features, keep BsmtFinType1 == "GLK", BsmtQual, BsmtCond, TotalBsmtSF
# sns.boxplot(x="BsmtFinType1", y="SalePrice", data=df_train)
df_train["BsmtFinTpGLQ"] = 0
df_train.loc[df_train.BsmtFinType1 == "GLQ", "BsmtFinTpGLQ"] = 1

df_test["BsmtFinTpGLQ"] = 0
df_test.loc[df_test.BsmtFinType1 == "GLQ", "BsmtFinTpGLQ"] = 1

bsmt_features = ["BsmtFinType2", "BsmtFinType1", "BsmtExposure", "BsmtFinSF1", "BsmtUnfSF"]
df_train.drop(bsmt_features, axis=1, inplace=True)
df_test.drop(bsmt_features, axis=1, inplace=True)

df_train.loc[:, ["BsmtCond", "BsmtQual"]] = df_train.loc[:, ["BsmtCond", "BsmtQual"]].fillna("No")
df_test.loc[:, ["BsmtCond", "BsmtQual"]] = df_test.loc[:, ["BsmtCond", "BsmtQual"]].fillna("No")


# MasVnr
df_train.MasVnrType.value_counts()  # None is most common by a lot
# sns.boxplot(x="MasVnrType", y="SalePrice", data=df_train) #There is difference between types
# sns.regplot("MasVnrArea", "SalePrice", data=df_train)
df_train[["MasVnrArea", "SalePrice"]].corr() #0.45

df_train["MasVnrType"].fillna("None", inplace=True)
df_test["MasVnrType"].fillna("None", inplace=True)

df_train["MasVnrArea"].fillna(0, inplace=True)
df_test["MasVnrArea"].fillna(0, inplace=True)

#Electrical
df_train.Electrical.fillna(df_train.Electrical.mode()[0], inplace=True)

#Fill missing values of cat data in test set
cat_cols = ["MSZoning", "Utilities", "Functional", "SaleType", "Exterior2nd", "Exterior1st", "KitchenQual"]
for col in cat_cols:
    df_test.loc[:, [col]] = df_test[col].fillna(df_train[col].mode()[0])

df_test["BsmtFullBath"].fillna(df_train["BsmtFullBath"].median(), inplace=True)
df_test["TotalBsmtSF"].fillna(df_train["TotalBsmtSF"].mean(), inplace=True)


# Change YearBuilt to Antiquity
df_train["Antiquity"] = 2010 - df_train["YearBuilt"]
df_test["Antiquity"] = 2010 - df_test["YearBuilt"]
df_train.drop(["YearBuilt"], axis=1, inplace=True)
df_test.drop(["YearBuilt"], axis=1, inplace=True)

#TODO scale : y too?
numeric = df_train.select_dtypes(include=[np.number]).drop(["SalePrice", "Garage", "BsmtFinTpGLQ"], axis=1)
numeric_test = df_test.select_dtypes(include=[np.number]).drop(["Garage", "BsmtFinTpGLQ"], axis=1)
category = df_train.select_dtypes(exclude=[np.number])
category_test = df_test.select_dtypes(exclude=[np.number])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_train = scaler.fit_transform(numeric)
scaled_test = scaler.transform(numeric_test)

df_train[list(numeric.columns.values)] = scaled_train
df_test[list(numeric.columns.values)] = scaled_test
#dummies
dummies = pd.get_dummies(category, drop_first=True)
dummies_test = pd.get_dummies(category_test, drop_first=True)
dummies_not_in_test = [ dummy for dummy in list(dummies.columns.values) if dummy not in list(dummies_test.columns.values)]
dummies.drop(dummies_not_in_test, axis=1, inplace=True)

df_train_prepared = df_train.join(dummies)
df_train_prepared.drop(list(category.columns.values), axis=1, inplace=True)
y = df_train.SalePrice
df_train_prepared.drop(["SalePrice"], axis=1, inplace=True)
df_test_prepared = df_test.join(dummies_test)
df_test_prepared.drop(list(category.columns.values), axis=1, inplace=True)


# TODO Feature selection


# TODO modelling
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import math
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def rmsle(y_true, y_pred):
    return math.sqrt(abs(mean_squared_log_error(y_true, y_pred)))
X_train, X_test, y_train, y_test = train_test_split(df_train_prepared, y, test_size=0.3)
predictors = [Ridge(),
              DecisionTreeRegressor(),
              SVR(),
              RandomForestRegressor(),
              AdaBoostRegressor(),
              GradientBoostingRegressor()]

for predictor in predictors:
    predictor.fit(X_train, y_train)
    print (predictor.__class__, rmsle(y_test, predictor.predict(X_test)))


# Using cross validation

from sklearn.model_selection import cross_val_score

def cv_score(clf, X, y, n_splits=5):
    # cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=n_splits, scoring='neg_mean_squared_log_error')
    print(clf.__class__, "srmle: %0.5f , rmle: %0.5f (+/- %0.5f)" % (math.sqrt(abs(scores.mean())), scores.mean(), scores.std() * 2))
    return abs(scores.mean())

def perform_grid_search(clf, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='neg_mean_squared_log_error')
    grid_search.fit(X, y)
    print ("Best params", grid_search.best_params_)
    return grid_search.best_estimator_

def score(predictor, X, y):
    msle = mean_squared_log_error(y, predictor.predict(X))
    print (math.sqrt(msle), msle)

for predictor in predictors:
    cv_score(predictor, df_train_prepared, y)


# Tuning hyperparams
from sklearn.model_selection import GridSearchCV
scores = {}
ridge_param_grid = {
    "alpha":[15,16, 17, 18, 19, 20] #Regularization strenght, higher alpha, more reg
}
ridge_regressor = perform_grid_search(Ridge(), ridge_param_grid, X_train, y_train)# df_train_prepared, y)
scores["ridge"] = cv_score(ridge_regressor, df_train_prepared, y)


tree_param_grid = {
    "max_depth": [5, 7, 10, 20, 50, 100]
}
tree_regressor = perform_grid_search(DecisionTreeRegressor(), tree_param_grid, df_train_prepared, y)
scores["tree"] = cv_score(tree_regressor, df_train_prepared, y)
score(tree_regressor, X_test, y_test)

rf_tree_param_grid = {
    "max_depth": [8, 10,15, 20, 50, 100],
    "n_estimators": [20, 50, 100, 150]
}
rf_tree_regressor = perform_grid_search(RandomForestRegressor(), rf_tree_param_grid, df_train_prepared, y)
scores["rf"] = cv_score(rf_tree_regressor, df_train_prepared, y)

ada_param_grid = {
    "n_estimators":[20, 50, 70, 100, 150],
    "learning_rate": [1, 10, 100, 0.1]
}
ada_regressor = perform_grid_search(AdaBoostRegressor(), ada_param_grid, df_train_prepared, y)
scores["ada"] = cv_score(ada_regressor, df_train_prepared, y)

gb_param_grid = {
    "n_estimators": [10, 20, 50, 100, 150],
    "max_depth": [5, 10, 20, 50, 100]
}
gb_regressor = perform_grid_search(GradientBoostingRegressor(), gb_param_grid, df_train_prepared, y)
scores["gb"] = cv_score(gb_regressor, df_train_prepared, y)

xgb_param_grid = {
    "n_estimators": [10, 20, 50, 100, 150],
    "max_depth": [5, 10, 20, 50, 100]
}
xgb_regressor = perform_grid_search(XGBRegressor(), xgb_param_grid, df_train_prepared, y)
scores["xgb"] = cv_score(xgb_regressor, df_train_prepared, y)


df = pd.read_csv('test.csv')
house_id = df.Id
def predict(clf, X_test, csv_name):
    global house_id
    predictions = clf.predict(X_test)
    pred_df = pd.DataFrame()
    pred_df["Id"] = house_id
    pred_df["SalePrice"] = predictions
    pred_df.to_csv(csv_name, index=False)

predict(rf_tree_regressor, df_test_prepared, "prediction_rf_v1.csv")
predict(xgb_regressor, df_test_prepared, "prediction_xgb_v2.csv")

predictors = [ridge_regressor, tree_regressor, rf_tree_regressor, ada_regressor, gb_regressor, xgb_regressor]
df = pd.DataFrame()
for i,predictor in enumerate(predictors):
    df[str(i)] = predictor.predict(df_test_prepared)


df['SalePrice'] = df.mean(axis=1)
df["Id"] = house_id
df[["Id", "SalePrice"]].to_csv("prediction_avg_v3.csv", index=False)


from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Lasso


stack = StackingCVRegressor(regressors = (ridge_regressor, tree_regressor, rf_tree_regressor, ada_regressor, gb_regressor, xgb_regressor), meta_regressor = Ridge())
cv_score(stack, df_train_prepared.values, y.values)

stack.fit(df_train_prepared.values, y.values)
predict(stack, df_test_prepared.values, "prediction_stack_v4.csv")


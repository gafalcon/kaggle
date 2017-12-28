import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib
df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")
y = df_train.SalePrice

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
sns.violinplot(x="FireplaceQu", y="SalePrice", data=df_train)

df_train.drop(["FireplaceQu"], axis=1, inplace=True)
df_test.drop(["FireplaceQu"], axis=1, inplace=True)

numeric = df_train.select_dtypes(include=[np.number])
category = df_train.select_dtypes(exclude=[np.number])


# Select columns with low correlation
cors = numeric.corr()["SalePrice"]
low_cor_columns = list(cors[abs(cors) < 0.2].index.values)
sns.pairplot(numeric[low_cor_columns[4:8]+ ["SalePrice"]], dropna=True)

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
sns.boxplot(x="BsmtFinType1", y="SalePrice", data=df_train)
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
sns.boxplot(x="MasVnrType", y="SalePrice", data=df_train) #There is difference between types
sns.regplot("MasVnrArea", "SalePrice", data=df_train)
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
df_test_prepared = df_test.join(dummies_test)
df_test_prepared.drop(list(category.columns.values), axis=1, inplace=True)


# TODO Feature selection


# TODO modelling

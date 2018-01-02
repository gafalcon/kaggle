# DONE Load datasets, merge them without SalePrice, variables id, and saleprice
# TODO find and delete outliers
# DONE impute missing data
# DONE transform skew data
# TODO scale numeric
# DONE dummies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib

df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

train_id = df_train.Id
test_id = df_test.Id

for df in [df_train, df_test]:
    df.drop(["Id"], axis=1, inplace=True)

all_data = pd.concat([df_train, df_test])

# Find and delete outliers
cat_cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold']

for col in cat_cols:
    all_data[col] = all_data[col].astype('category')

numeric = all_data.select_dtypes(include=[np.number]).columns

f,ax=plt.subplots(3,3)#,figsize=(18,18)) # Subplots
ax = ax.flatten()
for i, col in enumerate(numeric[18:27]):
    df_train.plot.scatter(x=col, y="SalePrice", ax=ax[i])


df_train.plot.scatter(x="GrLivArea", y="SalePrice")
sns.regplot("GrLivArea", "SalePrice", data=df_train)
#Deleting outliers
df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index, inplace=True)

# Transform skewed SalePrice
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
SalePrices = df_train.SalePrice
ntrain = df_train.shape[0]
ntest = df_test.shape[0]

# Missing data
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
def missing_data(train):
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data[missing_data["Total"] > 0]

print(missing_data(all_data))

missing_data_cols = ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType", "MSSubClass")

for col in missing_data_cols:
    all_data[col] = all_data[col].fillna("None")

missing_data_cols = ("MasVnrArea", "GarageArea", "GarageCars", "GarageYrBlt", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath")
for col in missing_data_cols:
    all_data[col] = all_data[col].fillna(0)

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data = all_data.drop(['Utilities'], axis=1)


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())) #According to violinplot of Neighborhood vs LotFrontage it doesnt seem to be the case


# Transform to categorical
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].astype("category")

#Changing OverallCond into a categorical variable
#all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype("category")
all_data['MoSold'] = all_data['MoSold'].astype("category")


#Label encoding categorical variables that can be ordinal
# TODO correct order
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit((all_data[c].values))
    all_data[c] = lbl.transform((all_data[c].values))


# Adding total sqfootage feature
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
from scipy.stats import skew
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)
print(all_data.shape)

train = all_data[:ntrain]
test = all_data[ntrain:]

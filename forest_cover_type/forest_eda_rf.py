
# coding: utf-8

# In[13]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import scipy
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("."))
PATH = "."
# Any results you write to the current directory are saved as output.


# In[14]:


df = pd.read_csv(f"{PATH}/train.csv")


# In[15]:


df.describe().T


# In[16]:


y = df.Cover_Type
df.drop(["Id", "Cover_Type"], axis=1, inplace=True)


# In[17]:


cols = df.columns.values


# In[18]:


rf_classifier = RandomForestClassifier()
rf_classifier.fit(df, y)


# In[19]:


rf_classifier.score(df, y)


# In[20]:


X_train, X_valid, y_train, y_valid = train_test_split(df, y, test_size=0.3, random_state = 7)
X_train = X_train.values
X_valid = X_valid.values
y_train = y_train.values
y_valid = y_valid.values


# In[21]:


unique, counts = np.unique(y_train, return_counts=True)

print (np.asarray((unique, counts)).T)


# In[22]:


def print_score(m):
    print ("train:",m.score(X_train,y_train), "valid:", m.score(X_valid, y_valid), "oob: ", m.oob_score_)


# ## Baseline Classifier

# In[23]:


rf_classifier = RandomForestClassifier(oob_score=True)
rf_classifier.fit(X_train, y_train)
print_score(rf_classifier)


# ## Impact of number of trees

# In[24]:


rf_classifier = RandomForestClassifier(n_estimators = 100, oob_score=True)
rf_classifier.fit(X_train, y_train)
print_score(rf_classifier)


# In[25]:


preds = np.stack([t.predict(X_valid) for t in rf_classifier.estimators_])


# In[26]:


preds[:,0], scipy.stats.mode(preds[:,0])[0], y_valid[0]


# In[27]:


plt.plot([accuracy_score(y_valid, scipy.stats.mode(preds[:i+1], axis=0)[0][0]+1) for i in range(len(rf_classifier.estimators_))]);


# ## Reducing Overfitting

# In[28]:


c = RandomForestClassifier(min_samples_leaf=3, n_estimators=100, max_features=0.5, oob_score=True)
c.fit(X_train, y_train)
print_score(c)


# ### Inspection

# In[29]:


cm = confusion_matrix(y_valid, c.predict(X_valid))


# In[30]:


sns.heatmap(cm, annot=True, annot_kws={"size": 8});


# ### Feature Importance

# In[31]:


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# In[32]:


fi = rf_feat_importance(c, df); fi[:15]


# In[33]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[34]:


plot_fi(fi[:30]);


# ## Adding Features

# In[35]:


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


# In[36]:


df = add_features(df)


# In[37]:


X_train, X_valid, y_train, y_valid = train_test_split(df, y, test_size=0.3, random_state = 7)
X_train.columns


# In[38]:


c = RandomForestClassifier(min_samples_leaf=3, n_estimators=100, max_features=0.5, oob_score=True)
c.fit(X_train, y_train)
print_score(c)


# In[39]:


fi = rf_feat_importance(c, X_train)


# In[40]:


fi[:20]


# In[41]:


plot_fi(fi[:30])


# In[42]:


to_keep = fi[fi.imp>0.005].cols; len(to_keep)


# In[43]:


df_keep = df[to_keep].copy()
X_train, X_valid, y_train, y_valid = train_test_split(df_keep, y, test_size=0.3, random_state = 7)


# In[44]:


c = RandomForestClassifier(min_samples_leaf=3, n_estimators=100, max_features=0.5, oob_score=True)
get_ipython().run_line_magic('time', 'c.fit(X_train, y_train)')
print_score(c)


# In[45]:


fi = rf_feat_importance(c, X_train)
plot_fi(fi)


# ## Creating test preds

# In[46]:


c = RandomForestClassifier(min_samples_leaf=3, n_estimators=100, max_features=0.5, oob_score=True)
get_ipython().run_line_magic('time', 'c.fit(df_keep, y)')
print_score(c)


# In[47]:


X_test = pd.read_csv(f"{PATH}/test.csv")


# In[48]:


X_test = add_features(X_test)
test = X_test[to_keep].copy()


# In[49]:


test_preds = c.predict(test)


# In[50]:


res = pd.DataFrame({"Id": X_test.Id, "Cover_Type": test_preds})


# In[51]:


res.to_csv(f"submission_rf_new_features.csv", index=False)


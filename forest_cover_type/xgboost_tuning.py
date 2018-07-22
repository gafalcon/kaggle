from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV, cross_val_score

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

df = pd.read_csv("train.csv")
df = add_features(df)
df.to_csv("train_extra_features.csv", index=False)
df = pd.read_csv("train_extra_features.csv")
test = pd.read_csv("test.csv")
test = add_features(test)
test.to_csv("test_extra_features.csv", index=False)
test = pd.read_csv("test_extra_features.csv")

y = df.Cover_Type
y = y - 1 #for xgb boost, classes must be in [0, num_class]
df.drop(["Id", "Cover_Type"], axis=1, inplace=True)

test_Ids = test.Id
test.drop(["Id"], axis=1, inplace=True)

def modelfit(alg, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(predictors.values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)#, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
        print (cvresult)
        print (cvresult.shape)
    # Fit the algorithm on the data
    alg.fit(predictors, target)#, eval_metric='auc')

    # # Predict training set
    train_preds = alg.predict(predictors)
    train_predprob = alg.predict_proba(predictors)[:,1]

    # #Model report
    print ("Accuracy: %.4g" % metrics.accuracy_score(target.values, train_preds))
    # # print ("AUC SCORE (Train): %f" % metrics.roc_auc_score(target, train_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title="Feature Importances")
    plt.ylabel('Feature Importance Score')

xgb1 = XGBClassifier(
    learning_rate = 0.5,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=4,
    scale_pos_weight=1,
    seed=7,
    objective='multi:softmax',
    num_class=7
    )

%time modelfit(xgb1, df, y) #si el n_estimators es muy grande, aumentar lr, y repetir para encontrar optimo n_estimator
# n_estimators: 300, lr: 0.5

param_test1 = {
 'max_depth':range(3,15,2),
 #'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.5, n_estimators=300, max_depth=5,
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', num_class=7, nthread=4, scale_pos_weight=1,
                                                  seed=27),  param_grid = param_test1, n_jobs=4,iid=False, cv=5)
%time gsearch1.fit(df, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

#([mean: 0.78578, std: 0.03423, params: {'max_depth': 3},
#  mean: 0.80344, std: 0.03430, params: {'max_depth': 5},
#  mean: 0.80615, std: 0.03695, params: {'max_depth': 7},
#  mean: 0.80926, std: 0.03207, params: {'max_depth': 9},
#  mean: 0.80747, std: 0.03296, params: {'max_depth': 11},
#  mean: 0.80661, std: 0.03418, params: {'max_depth': 13}],
# {'max_depth': 9},
# 0.8092592592592593)

param_test1 = {
 'max_depth':range(9,15,1),
 #'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.5, n_estimators=300, max_depth=5,
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', num_class=7, nthread=4, scale_pos_weight=1,
                                                  seed=27),  param_grid = param_test1, n_jobs=4,iid=False, cv=5)
%time gsearch1.fit(df, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

#([mean: 0.80787, std: 0.03269, params: {'max_depth': 9},
#  mean: 0.80681, std: 0.03440, params: {'max_depth': 10},
#  mean: 0.81085, std: 0.03320, params: {'max_depth': 11},
#  mean: 0.80952, std: 0.03437, params: {'max_depth': 12},
#  mean: 0.81230, std: 0.03406, params: {'max_depth': 13},
#  mean: 0.81065, std: 0.03402, params: {'max_depth': 14}],
# {'max_depth': 13},
# 0.8123015873015873)

param_test_gamma = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.5, n_estimators=300, max_depth=9,
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', num_class=7, nthread=4, scale_pos_weight=1,
                                                  seed=27),  param_grid = param_test_gamma, n_jobs=4,iid=False, cv=5)
%time gsearch1.fit(df, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#([mean: 0.80926, std: 0.03207, params: {'gamma': 0.0},
#  mean: 0.80483, std: 0.03337, params: {'gamma': 0.1},
#  mean: 0.80245, std: 0.03605, params: {'gamma': 0.2},
#  mean: 0.80271, std: 0.03552, params: {'gamma': 0.3},
#  mean: 0.79888, std: 0.03531, params: {'gamma': 0.4}],
# {'gamma': 0.0},
# 0.8092592592592593)

 #a good idea would be to re-calibrate the number of boosting rounds for the updated parameters.
 xgb1 = XGBClassifier(
    learning_rate = 0.5,
    n_estimators=1000,
    max_depth=13,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=4,
    scale_pos_weight=1,
    seed=7,
    objective='multi:softmax',
    num_class=7
    )

%time modelfit(xgb1, df, y)


param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.5, n_estimators=300, max_depth=9,
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', num_class=7, nthread=4, scale_pos_weight=1,
                                                  seed=27),  param_grid = param_test4, n_jobs=4,iid=False, cv=5)
%time gsearch1.fit(df, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

#
#([mean: 0.80304, std: 0.03498, params: {'colsample_bytree': 0.6, 'subsample': 0.6},
#  mean: 0.80324, std: 0.03119, params: {'colsample_bytree': 0.6, 'subsample': 0.7},
#  mean: 0.80754, std: 0.03197, params: {'colsample_bytree': 0.6, 'subsample': 0.8},
#  mean: 0.80933, std: 0.03472, params: {'colsample_bytree': 0.6, 'subsample': 0.9},
#  mean: 0.80251, std: 0.03366, params: {'colsample_bytree': 0.7, 'subsample': 0.6},
#  mean: 0.80595, std: 0.03380, params: {'colsample_bytree': 0.7, 'subsample': 0.7},
#  mean: 0.80556, std: 0.03340, params: {'colsample_bytree': 0.7, 'subsample': 0.8},
#  mean: 0.80999, std: 0.03499, params: {'colsample_bytree': 0.7, 'subsample': 0.9},
#  mean: 0.80423, std: 0.03528, params: {'colsample_bytree': 0.8, 'subsample': 0.6},
#  mean: 0.80556, std: 0.03387, params: {'colsample_bytree': 0.8, 'subsample': 0.7},
#  mean: 0.80926, std: 0.03207, params: {'colsample_bytree': 0.8, 'subsample': 0.8},
#  mean: 0.81177, std: 0.03377, params: {'colsample_bytree': 0.8, 'subsample': 0.9},
#  mean: 0.80212, std: 0.03277, params: {'colsample_bytree': 0.9, 'subsample': 0.6},
#  mean: 0.80430, std: 0.03301, params: {'colsample_bytree': 0.9, 'subsample': 0.7},
#  mean: 0.80708, std: 0.03343, params: {'colsample_bytree': 0.9, 'subsample': 0.8},
#  mean: 0.80800, std: 0.03514, params: {'colsample_bytree': 0.9, 'subsample': 0.9}],
# {'colsample_bytree': 0.8, 'subsample': 0.9},
# 0.8117724867724869)

param_test5 = {
 'subsample':[i/100.0 for i in range(85,100,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.5, n_estimators=300, max_depth=9,
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', num_class=7, nthread=4, scale_pos_weight=1,
                                                  seed=27),  param_grid = param_test5, n_jobs=4,iid=False, cv=5)
%time gsearch1.fit(df, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

#([mean: 0.80820, std: 0.03260, params: {'colsample_bytree': 0.75, 'subsample': 0.85},
#  mean: 0.80820, std: 0.03282, params: {'colsample_bytree': 0.75, 'subsample': 0.9},
#  mean: 0.80820, std: 0.03308, params: {'colsample_bytree': 0.75, 'subsample': 0.95},
#  mean: 0.80966, std: 0.03443, params: {'colsample_bytree': 0.8, 'subsample': 0.85},
#  mean: 0.81177, std: 0.03377, params: {'colsample_bytree': 0.8, 'subsample': 0.9},
#  mean: 0.80860, std: 0.03484, params: {'colsample_bytree': 0.8, 'subsample': 0.95},
#  mean: 0.80840, std: 0.03525, params: {'colsample_bytree': 0.85, 'subsample': 0.85},
#  mean: 0.80979, std: 0.03218, params: {'colsample_bytree': 0.85, 'subsample': 0.9},
#  mean: 0.80999, std: 0.03551, params: {'colsample_bytree': 0.85, 'subsample': 0.95}],
# {'colsample_bytree': 0.8, 'subsample': 0.9},
# 0.8117724867724869)


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.5, n_estimators=300, max_depth=9,
                                                  min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', num_class=7, nthread=4, scale_pos_weight=1,
                                                  seed=27),  param_grid = param_test6, n_jobs=4,iid=False, cv=5)
%time gsearch1.fit(df, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#([mean: 0.81177, std: 0.03253, params: {'reg_alpha': 1e-05},
#  mean: 0.80933, std: 0.03305, params: {'reg_alpha': 0.01},
#  mean: 0.80787, std: 0.03134, params: {'reg_alpha': 0.1},
#  mean: 0.80675, std: 0.03394, params: {'reg_alpha': 1},
#  mean: 0.68684, std: 0.03379, params: {'reg_alpha': 100}],
# {'reg_alpha': 1e-05},
# 0.8117724867724867)

param_test6 = {
 'reg_alpha':[1e-5, 0]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.5, n_estimators=300, max_depth=9,
                                                  min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.8,
                                                  objective= 'multi:softmax', num_class=7, nthread=4, scale_pos_weight=1,
                                                  seed=27),  param_grid = param_test6, n_jobs=4,iid=False, cv=5)
%time gsearch1.fit(df, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#([mean: 0.81177, std: 0.03253, params: {'reg_alpha': 1e-05},
#  mean: 0.81177, std: 0.03377, params: {'reg_alpha': 0}],
# {'reg_alpha': 0},
# 0.8117724867724869)

 xgb1 = XGBClassifier(
    learning_rate = 0.1,
    n_estimators=1000,
    max_depth=13,
    min_child_weight=1,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.8,
    nthread=4,
    scale_pos_weight=1,
    seed=7,
    objective='multi:softmax',
    num_class=7,
    reg_alpha=0
    )

%time modelfit(xgb1, df, y)

def cv_score(clf, X, y, n_splits=5, scoring=None):
    # cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=n_splits, scoring=scoring)
    #print ("Scores with C=",C, scores)
    print("Scores: ", scores)
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    
%time cv_score(xgb1, df, y)

#Scores:  [0.81084656 0.79298942 0.7906746  0.80621693 0.87962963]
#Accuracy: 0.81607 (+/- 0.06537)
#Wall time: 4min 20s

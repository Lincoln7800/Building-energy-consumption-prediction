import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import scipy as sp
import sklearn
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import time


# 1.读取数据
train = pd.read_csv('train.csv')
validate = pd.read_csv('validate.csv')
#train = train[(~train['BuildingID'].isin([13,19]))].reset_index(drop=True)
#validate = validate[(~validate['BuildingID'].isin([13,19]))].reset_index(drop=True)

# print(train.head())

# 2.标准化
XX_elect_train = train.drop(['room','num','date','season','Energy used','humidity', 'Sunshine hours'], axis = 1)
XX_elect_test = validate.drop(['room','num','date','season','Energy used','humidity', 'Sunshine hours'], axis = 1)

#YY_elect_train = train['Recordper10wArea']
#YY_elect_test = validate['Recordper10wArea']
YY_elect_train = train['Energy used']
YY_elect_test = validate['Energy used']
print(XX_elect_train.head())
#print(XX_elect_test.head())
# print(YY_elect_train.head())
# print(YY_elect_test.head())

# # 3.Find the optimal number of trees.
#scores = pd.DataFrame()
#t0 = time.time()
#for n in range(1,4):
    #RF = RandomForestRegressor(n_estimators=n, max_depth=None, min_samples_split=2, random_state=0)
    #score = cross_val_score(RF, XX_elect_train, YY_elect_train,cv=10)
    #scores[n] = score
    #print(score)

#
# #sns.set_context("talk")
# #sns.set_style("white")
# sns.boxplot(data=scores)
# plt.xlabel("Number of trees")
# plt.ylabel("Scores")
# plt.title("The scores of the Random Forests for different number of trees.")
# #plt.xlim(0,41)
# plt.show()

# 4.使用随机森林训练预测
RF_e = RandomForestRegressor(n_estimators=40, max_depth=None, random_state=0)
t0 = time.time()
RF_e.fit(XX_elect_train, YY_elect_train)
print('训练时间：', time.time() - t0)
YY_elect_pred=RF_e.predict(XX_elect_test)

#计算Feature importances
importances = RF_e.feature_importances_
print(importances)
f_values = importances
f_index =  [ 'Average temperature','Max temperature','Min temperature', 'category','direction','floor number', 'Number of households','Student attributes']
x_index = list(range(0,8))
x_index = [x/0.6 for x in x_index]
plt.rcParams['figure.figsize'] = (17,10)
plt.barh(x_index, f_values, tick_label=f_index)
plt.xlabel('importances',fontsize = 15)
plt.ylabel('features',fontsize= 15)
plt.show()

# 5.计算CV-RMSE
# YY_elect_pred = YY_elect_pred * validate['Area'] / 100000
YY_elect_test = validate['Energy used']
MSE = mean_squared_error(YY_elect_test, YY_elect_pred)
print('MSE', MSE)
RMSE = np.sqrt(MSE)
mean = np.mean(YY_elect_test)
print('CV-RMSE', RMSE/mean)

# 6.画图
# fig,ax = plt.subplots(1, 1,figsize=(20,10))
# line1, =plt.plot(XX_elect_test.index, YY_elect_test, label='Actual consumption', color='k')
# line2, =plt.plot(XX_elect_test.index, YY_elect_pred, label='RF Regression Prediction', color='r')
# plt.xlabel('Feature index',fontsize=18)
# plt.ylabel('W (kWh)',fontsize=18)
# plt.title('Actual and RF predicted W',fontsize=20)
# plt.legend([line1, line2], ['Actual consumption', 'RF Regression Prediction'],fontsize=18)
# plt.show()
#
# #Plot actual vs. prediced usage.
# fig = plt.figure(figsize=(8,8))
# plt.scatter(YY_elect_test, YY_elect_test, c='k')
# plt.scatter(YY_elect_test, YY_elect_pred, c='r')
# plt.xlabel('Actual W (kWh): $Y_i$',fontsize=18)
# plt.ylabel("Predicted W (kWh): $\hat{Y}_i$",fontsize=18)
# plt.title("Actual W vs Predicted W: $Y_i$ vs $\hat{Y}_i$",fontsize=20)
# plt.show()

# # 7.导出预测数据
#YY_elect_pred = pd.DataFrame(YY_elect_pred,columns=['W'])
#YY_elect_pred.to_csv('Pred_RF_W.csv')

# 8.预测最终建筑
#Prefeatures = pd.read_csv('Prefeatures(2).csv').drop(['time','HourinYear','Area'], axis = 1)
#PreResult_Q=RF_e.predict(Prefeatures)
#PreResult_Q = pd.DataFrame(PreResult_Q,columns=['Q'])
#PreResult_Q.to_csv('PreResult_RF_Q（4.1.2）.csv')
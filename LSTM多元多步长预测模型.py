import numpy as np
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import metrics
from numpy import cov
import math

def Create_dataset(dataset, look_back):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        data_X.append(a)
        data_Y.append(dataset[i + look_back])
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    return data_X, data_Y
# 转换序列成监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def Normalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i] - low) / delta
    return list, low, high
def Normalize2(list,low,high):
    list = np.array(list)
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return  list

def FNoramlize(list, low, high):
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = list[i] * delta + low
    return list
def MBE(y_true, y_pred):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # y_true = y_true.reshape(len(y_true),1)
    # y_pred = y_pred.reshape(len(y_pred),1)
    diff = (y_true - y_pred)
    mbe = diff.mean()
    return(mbe)


def calc_corr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq

    return corr_factor


# 加载数据集
dataset = read_csv('W12.8.csv', header=0, index_col=0)
dataset.drop('dry-difference',axis= 1 )
values = dataset.values
# 整数编码
#encoder = LabelEncoder()
#values[:, 1] = encoder.fit_transform(values[:, 1])
# ensure all data is float
values = values.astype('float32')
reframed = series_to_supervised(values, 1, 1)
print(reframed.head())
values = reframed.values
n_train_hours = 365*2*24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 分为输入输出

# 归一化特征
train_n,train_low,train_high = Normalize(train)
#更新的归一化
test_n = Normalize2(test,train_low,train_high)
print(train_n,test_n)
# 构建监督学习问题
# 丢弃我们并不想预测的列
#reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

# 分割为训练集和测试集
train_X, train_y = train_n[:, :-1], train_n[:, -1]
test_X, test_y = test_n[:, :-1], test_n[:, -1]
# 重塑成3D形状 [样例, 时间步, 特征]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
data = pd.DataFrame()

model = Sequential()
model.add(LSTM(400, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
    # 拟合神经网络模型
history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
    # 绘制历史数据
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

    # 做出预测

train_predict = model.predict(train_X)
test_predict  = model.predict(test_X)
train_Y = FNoramlize(train_y,train_low,train_high)
train_predict = FNoramlize(train_predict,train_low,train_high)
test_Y = FNoramlize(test_y,train_low,train_high)
test_predict = FNoramlize(test_predict,train_low,train_high)
    #计算cv-RMSE
MBE = MBE(test_Y, test_predict)
print('MBE',MBE)
MSE = mean_squared_error(test_Y, test_predict)
RMSE = np.sqrt(MSE)

mean = np.mean(test_Y)
print('RMSE', RMSE)
    # MAE
print('MAE',metrics.mean_absolute_error(test_Y, test_predict)) # 1.9285714285714286
    # MAPE
# def mape(y_true, y_pred):
#     return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
# print('MAPE',mape(test_Y, test_predict))
    # covariance
COV = calc_corr(test_Y, test_predict)
print('COV',COV)

# data['a'] = MSE,RMSE,COV
# data.to_excel('7.17.xlsx',encoding='gbk',index= 0 )




# plt.subplot(121)
# plt.plot(train_Y)
# plt.plot(train_predict)
# plt.subplot(122)
# plt.plot(test_Y)
# plt.plot(test_predict)
# plt.show()

# new_data = pd.DataFrame()
# # date = test[:, 1]
# # new_data['date'] = date
# new_data['original data'] = test_Y
# new_data['predict data'] = test_predict
# new_data.to_csv('照明预测数据9.csv',encoding='gbk')

#PreResult_Q = pd.DataFrame(test_predict,columns=['预测'])
#PreResult_Q.to_csv('Result_W（1）.csv')

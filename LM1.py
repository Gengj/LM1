# 导入类库
import tensorflow

import numpy as np

from numpy import arange

from matplotlib import pyplot

from pandas import read_csv

from pandas import  set_option

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error

# 导入数据

filename = 'housing.data'

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',

         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']

dataset = read_csv(filename, names=names, delim_whitespace=True)

# 数据维度
print(dataset.shape)

# 特征属性的字段类型
print(dataset.dtypes)

# 查看最开始的30条记录
set_option('display.line_width', 200)
#
print(dataset.head(30))

# 描述性统计信息
set_option('precision', 1)

print(dataset.describe())

# 关联关系
set_option('precision', 2)

print(dataset.corr(method='pearson'))

# 直方图
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)

pyplot.show()

# 密度图
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, fontsize=1)

pyplot.show()

#箱线图

dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)

pyplot.show()

# 散点矩阵图

scatter_matrix(dataset)

pyplot.show()

# 相关矩阵图

fig = pyplot.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')

fig.colorbar(cax)

ticks = np.arange(0, 14, 1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

# ax.set_yticklabels(names)

pyplot.show()
import pandas as pd
import numpy as np
import statsmodels.api as sm


data = sm.datasets.macrodata.load_pandas().data
df = data[["realgdp"]]
#print(df)

#we take last 60 quarters data for each time-series point, i.e. lag=60
SIZE = 60 
COLUMNS = ['t{}'.format(x) for x in range(SIZE)] + ['realgdp']
list_train = []
for i in range(SIZE, df.shape[0]):
    list_train.append(df.loc[i-SIZE:i, 'realgdp'].tolist())
df_train = pd.DataFrame(list_train, columns=COLUMNS)
#print(df_train)

'''With this, we transform time series data line with length N into a data frame (table)
with (N-M) rows and M columns. Where M is our chosen length of past data points to use 
for each training sample (60 points(quarters) = 15 years in the example above)'''
df_feats = pd.DataFrame()
df_feats['prev_1'] = df_train.iloc[:,-2] #Here -2 as -1 is a target

for win in [2, 3, 5, 7, 10, 16, 20, 28, 56]:
    tmp = df_train.iloc[:,-1-win:-1]
    #General statistics for base level
    df_feats['mean_prev_{}'.format(win)] = tmp.mean(axis=1)
    df_feats['median_prev_{}'.format(win)] = tmp.median(axis=1)
    df_feats['min_prev_{}'.format(win)] = tmp.min(axis=1)
    df_feats['max_prev_{}'.format(win)] = tmp.max(axis=1)
    df_feats['std_prev_{}'.format(win)] = tmp.std(axis=1)
    #Capturing trend
    df_feats['mean_ewm_prev_{}'.format(win)] = tmp.T.ewm(com=9.5).mean().T.mean(axis=1)
    df_feats['last_ewm_prev_{}'.format(win)] = tmp.T.ewm(com=9.5).mean().T.iloc[:,-1]    
    df_feats['avg_diff_{}'.format(win)] = (tmp - tmp.shift(1, axis=1)).mean(axis=1)
    df_feats['avg_div_{}'.format(win)] = (tmp / tmp.shift(1, axis=1)).mean(axis=1)
print(df_feats)
""" for win in [4, 5, 7, 14]:
    tmp = df_train.iloc[:,-1-win*4:-1:4] #4 quarters for a year
    #Features for yearly seasonality
    df_feats['year_mean_prev_{}'.format(win)] = tmp.mean(axis=1)
    df_feats['year_median_prev_{}'.format(win)] = tmp.median(axis=1)
    df_feats['year_min_prev_{}'.format(win)] = tmp.min(axis=1)
    df_feats['year_max_prev_{}'.format(win)] = tmp.max(axis=1)
    df_feats['year_std_prev_{}'.format(win)] = tmp.std(axis=1)
 """
import lightgbm as lgb
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting': 'gbdt',
    'learning_rate': 0.06,
    'num_leaves': 64,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.9,
	'force_col_wise': True
}
x_train = lgb.Dataset(df_feats, df_train['realgdp'])
model = lgb.train(params, x_train, num_boost_round=500)
preds = model.predict(df_feats)
print(preds)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1) where n is the lag
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


raw = pd.DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(values, 1, 2)
print(data)
"""
Neural net implementation of electric load forecasting.
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.stats import zscore

# NERC6 holidays with inconsistent dates. Created with python holidays package
# years 1990 - 2024
with open('holidays.pickle', 'rb') as f:
	nerc6 = pickle.load(f)


def MAPE(predictions, answers):
		assert len(predictions) == len(answers)
		return sum([abs(x-y)/(y+1e-5) for x, y in zip(predictions, answers)])/len(answers)*100 


def isHoliday(holiday, df):
	# New years, memorial, independence, labor day, Thanksgiving, Christmas
	m1 = None
	if holiday == "New Year's Day":
		m1 = (df["dates"].dt.month == 1) & (df["dates"].dt.day == 1)
	if holiday == "Independence Day":
		m1 = (df["dates"].dt.month == 7) & (df["dates"].dt.day == 4)
	if holiday == "Christmas Day":
		m1 = (df["dates"].dt.month == 12) & (df["dates"].dt.day == 25)
	m1 = df["dates"].dt.date.isin(nerc6[holiday]) if m1 is None else m1
	m2 = df["dates"].dt.date.isin(nerc6.get(holiday + " (Observed)", []))
	return m1 | m2


def add_noise(m, std):
	noise = np.random.normal(0, std, m.shape[0])
	return m + noise


def makeUsefulDf(df, noise=2.5, hours_prior=24):
	"""
	Turn a dataframe of datetime and load data into a dataframe r_df 
	useful for machine learning. Normalize values.
	"""	
	if 'dates' not in df.columns:
		df['dates'] = df.apply(lambda x: dt(int(x['year']), int(x['month']), int(x['day']), int(x['hour'])), axis=1)

	r_df = pd.DataFrame()
	
	# LOAD and Normalize, column load_prev_n represents 24 hr before load
	r_df["load_n"] = zscore(df["load"])
	r_df["load_prev_n"] = r_df["load_n"].shift(hours_prior)
	r_df["load_prev_n"].bfill(inplace=True)
	
	# LOAD PREV
	def _chunks(l, n):
		#slice df rows by each n periods (24 hours in this case)
		return [l[i : i + n] for i in range(0, len(l), n)]

	n = np.array([val for val in _chunks(list(r_df["load_n"]), 24) for _ in range(24)])
	l = ["l" + str(i) for i in range(24)]

	for i, s in enumerate(l):
		r_df[s] = n[:, i]
		r_df[s] = r_df[s].shift(hours_prior)
		r_df[s] = r_df[s].bfill()
	r_df.drop(['load_n'], axis=1, inplace=True)
	
	# DATE
	r_df["years_n"] = zscore(df["dates"].dt.year)
	r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.hour, prefix='hour')], axis=1)
	r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.dayofweek, prefix='day')], axis=1)
	r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.month, prefix='month')], axis=1)
	for holiday in ["New Year's Day", "Memorial Day", "Independence Day", "Labor Day", "Thanksgiving", "Christmas Day"]:
		r_df[holiday] = isHoliday(holiday, df)

	# TEMP
	temp_noise = df['tempc'] + np.random.normal(0, noise, df.shape[0])
	r_df["temp_n"] = zscore(temp_noise)
	r_df['temp_n^2'] = zscore([x*x for x in temp_noise])
	return r_df


def data_transform(data, window, var='x'):
    m = []  
    for i in range(data.shape[0]-window):  # starting index of a day sample
        m.append(data[i:i+window].tolist())
    if var == 'x':
        t = np.zeros((len(m), len(m[0]), len(m[0][0])))
        for i, x in enumerate(m):  # x is each day sample
            for j, y in enumerate(x):  # y is each hour sample
                for k, z in enumerate(y):  # z is each feature
                    t[i, j, k] = z
    else:
        t = np.zeros((len(m), len(m[0])))
        for i, x in enumerate(m):
            for j, y in enumerate(x):
                t[i, j] = y
    return t


def hour_ahead_predictions(all_X, all_y, EPOCHS=10):
	from tensorflow.keras.models import Sequential, load_model
	from tensorflow.keras.layers import Dense
	from tensorflow import keras
	# slice ndarray to only leave last year(8760 hrs) as testing set
	X_train, y_train = all_X[:-8760, :], all_y[:-8760]
	X_test, y_test = all_X[-8760:, :], all_y[-8760:]
	input_shape = X_train.shape[1] 
	model = Sequential()
	model.add(Dense(input_shape, input_dim=input_shape, activation='relu'))
	model.add(Dense(input_shape, activation='relu'))
	model.add(Dense(input_shape, activation='relu'))
	model.add(Dense(input_shape, activation='relu'))
	model.add(Dense(input_shape, activation='relu'))
	model.add(Dense(1))
	
	optimizer = keras.optimizers.RMSprop(0.0001)
	model.compile(
		loss="mean_squared_error",
		optimizer=optimizer,
		metrics=["mean_absolute_error", "mean_squared_error"],
	)
	early_stop = keras.callbacks.EarlyStopping(monitor="mean_absolute_error", patience=20)
	history = model.fit(
		X_train,
		y_train,
		epochs=EPOCHS,
		verbose=1,
		callbacks=[early_stop],
	)
	predictions = [float(f) for f in model.predict(X_test)]
	train = [float(f) for f in model.predict(X_train)]
	accuracy = {
		'test': MAPE(predictions, y_test),
		'train': MAPE(train, y_train)
	}
	#save the model
	model.save('./models/hour_ahead_forecastor.h5')
	return predictions, accuracy


def day_ahead_predictions(all_X, all_y, window, EPOCHS=10):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow import keras
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping	
    # slice ndarray to only leave last year(8760 hrs) as testing set
    X_train, y_train = all_X[:-8760, :, :], all_y[:-8760, :]
    X_test, y_test = all_X[-8760:, :, :], all_y[-8760:, :]
    # build ANN(MLP) model, instead of calculating a single hour, combine 
    # all weights into one flat, fully-connected dense layer (half of window
    # * features number of nodes). That layer is then fully connected to a 
    # period vector for period ahead forecast.
    s = all_X.shape[2]
    model = Sequential()
    model.add(Dense(s, activation="relu", input_shape=(window, s)))
    model.add(Dense(s, activation="relu"))
    model.add(Dense(s, activation="relu"))
    model.add(Dense(s, activation="relu"))
    model.add(Dense(s, activation="relu"))
    model.add(Flatten())
    model.add(Dense(s*window//2, activation="relu"))
    model.add(Dense(window))
    # define customized optimizer
    nadam = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=nadam, loss='mape', metrics=['mae', 'mape'])	

    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
                 EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
    )

    predictions = np.array([])
    for row in X_test:
        X_test_row = row.reshape(1, window, s)
        yhat = model.predict(X_test_row, verbose=0)
        predictions = np.append(predictions, yhat)

    train = np.array([])
    for row in X_train:
        X_train_row = row.reshape(1, window, s)
        ytrainhat = model.predict(X_train_row, verbose=0)
        train = np.append(train, ytrainhat)

    y_test_flatten = y_test.flatten()
    y_train_flatten = y_train.flatten()

    accuracy = {
        'test': MAPE(predictions, y_test_flatten),
        'train': MAPE(train, y_train_flatten)
    }
	# save the model
    model.save('./models/day_ahead_forecastor.h5')    
    return predictions, accuracy
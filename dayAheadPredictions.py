from glob import glob
import logging, os
import pandas as pd
import numpy as np
import math
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import save_model
from keras.models import load_model


#INPUTS: names of csv files, name of zone being examined
#read in all csv files at once from directory
#feed in your filenames here for HOURLY LBMP
path = "./2017_NYISO_LBMPs/20170101damlbmp_zone_csv"
filenames = glob(os.path.join(path, "201701*.csv"))
#get list of dataframes for each day
dataframes= [pd.read_csv(f, header= 0, index_col=0) for f in filenames]
#input ZONE name here
zone = 'N.Y.C.'

num = len(dataframes)


df1 = [frame['LBMP ($/MWHr)'] for frame in dataframes]
df = pd.concat(df1)
    

#read in our dataframe for individual stock
# df = df['LBMP ($/MWHr)']
# df = pd.DataFrame(data= df)
df.columns = ['LBMP ($/MWHr)']
#setting our target variables
target1 = ['LBMP ($/MWHr)']  # this will have to be changed as our inputs change
lastIndex = len(df['LBMP ($/MWHr)'])  # get last index of dataframe
numForecastDays = 24  # variable for number of hours we want to forecast out
forecastDays_Index = lastIndex - numForecastDays  # index to take hours we want to forecast out off of datafram

#######################################
# Specify number of lag days, number of features and number of hours we are forecasting out
n_days = 1
n_features = len(df.columns)  # got from varNo in view of reframed Dataframe
forecast_out = numForecastDays

prevDay = df.iloc[forecastDays_Index-1]
prevDay = prevDay['LBMP ($/MWHr)']
prevDay_array = [prevDay]

#recording days to input to make our prediction
lastVals = df.iloc[(forecastDays_Index-1-n_days):(forecastDays_Index-1)]
daysList = []
if x == 1:
    daysList = [lastVals]
else:
    daysList.append(lastVals)


#recording the true price values for the days we are predicting on
trueValues = df.iloc[(forecastDays_Index):]
trueValues = trueValues['LBMP ($/MWHr)'].values
trueValues = trueValues.reshape(1,numForecastDays)
dfTrueVals = pd.DataFrame(data= trueValues)

###############################################################################
#our differencing to make predictions stationary
df = -df.diff()
df = df.drop(df.index[0]) #drop our first row of nans
df = df.drop(df.index[forecastDays_Index:lastIndex])  # drop the days that we are going to predict out on
values = df.values  # convert out dataframe to array of numpy values for our calculations


# convert series to supervised learning
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

# integer encode direction
# ensure all data is float
values = values.astype('float32')
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # Normalizing our data
values = scaler.fit_transform(values)

#specify the name of our target variable
#target = 'var' + str(y) + '(t)'
target = [] #create an empty list to append our target names
y = 1 #variable to iterate over in over
for x in range(1,len(df.columns)+1):
    target.append('var' + str(y) + '(t)')
    y +=1
    #print(target)

# frame as supervised learning
reframed = series_to_supervised(values, n_days, 1)
b = 0
for x in range(1,n_features+1):
    named = 'var' + str(x) + '(t)'
    targetName = target[b]
    b += 1
    if named != targetName:
        reframed = reframed.drop(named, axis = 1)

#print('REFRAMED:', reframed)
values = reframed.values  # convert reframed dataframe into numpy array

# split into train and test sets
n_train_days = int(0.8 * len(reframed['var1(t-1)']))  # using 80% of our data as our training set
n_test_days = int(len(reframed['var1(t-1)']) - n_train_days)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-n_features], train[:, -n_features:]
test_X, test_y = test[:, :-n_features], test[:, -n_features:]
# reshape to be 3d input [samples, timesteps, features]
# samples are the number of training/ tesing days
train_X = train_X.reshape(n_train_days, n_days, n_features)
test_X = test_X.reshape(n_test_days, n_days, n_features)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
# input shape has dimesions (time step, features)
numEpochs = 50
numBatch = 24
model.add(LSTM(50, input_shape=(n_days, n_features)))  # feeding in 1 time step  with 7 features at a time
model.add(Dense(n_features))
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])

# fit network
history = model.fit(train_X, train_y, epochs=numEpochs, batch_size=numBatch, validation_data=(test_X, test_y), verbose=0,
                        shuffle=False)

filename = "model" + str(1)+'.h5'
model.save(filename)
model = load_model(filename)
model.fit(test_X, test_y, epochs = numEpochs, batch_size = numBatch, verbose = 2, shuffle = False) #add state back in, or differencing technique
model.save(filename)

a = 0 #iterator for changing name of predictions for each stock exported to CSV
y = 1#iterator for dfName only
c = 0 #iterator for prevDay array


#########################################################################################################

#m = 0 #initialize variable to iterate through our dataframes in loop
k = 0 #iterate for string names

#initializing a list to store our lastVals dataframes in
r = 1 #iterator to keep track of lastVals dataframes

prevDay_array = np.zeros(num)

###############################################################


q = 1 #iterator for lastVals names

for x in range(1,2):
    modelName = "model" + str(1)+'.h5'
    print(modelName + 'running')
    model = load_model(modelName)
    # make a prediction
    num2 = x - 1 #the first index for lastVals
    dfLastValues = daysList[num2]
    lastVals = dfLastValues.values #turn our dataframe into an array
    #lastVals = df_lastVals.iloc[num2].values
    lastVals = lastVals.reshape((1,n_days,n_features))
    #new = test_X[-1].reshape((1,1,n_features))
    #yhat = model.predict(test_X[-1])

    #make prediction
    yhat = model.predict(lastVals)
    df_Prediction = pd.DataFrame(data = yhat)
    lastVals = df.iloc[(forecastDays_Index - n_days):(forecastDays_Index)].values
    lastVals = np.vstack((lastVals,yhat))
    prevDay_array2 = int(prevDay_array[c])


    #our loop within a loop for multiple forecasting days

    predictedPrice_array = np.zeros(numForecastDays)
    errorArray = np.zeros(numForecastDays)


    for x in range(1,numForecastDays+1):
        #get the predictions for change in magnitude of price
        predVals = lastVals[-n_days:] #slice so we are predicting on the last given days
        predVals = predVals.reshape((1,n_days,n_features))
        new_yhat = model.predict(predVals)#predicing next day out values
        # store our previous predicted values for sliding window
        lastVals = np.vstack((lastVals, new_yhat))  # keep stacking our output arrays
        lastVals = lastVals[-n_days:]  # slice so that we get out our

        #invert our normalized values
        invertVals = scaler.inverse_transform(new_yhat)
        df_InvertedVals = pd.DataFrame(data=invertVals)
        df_InvertedVals.columns = ['LBMP']
        #df_InvertedVals.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        #df_InvertedVals = df_InvertedVals.drop(df_InvertedVals.index[0]) #drop our first row of datafram
        predictedVals = df_InvertedVals['LBMP'].values  # Get out predicted values into a dataframe

        if x == 1:
            df_predictedVals = pd.DataFrame(data=predictedVals)
        else:
            df1 = pd.DataFrame(data= predictedVals)
            df_predictedVals = df_predictedVals.append(df1)



    #############################################
    df_predictedVals.reset_index(drop = True)
    trueVals = dfTrueVals.iloc[num2].values
    #trueVals = dfTrueVals.values
    #create an empty array to fill with rmse for each data point (to record how error is as time goes on)
    predictedVals = df_predictedVals.values #create an array of our predicted stock price changes
    df_predictedVals.reset_index(drop=True)

    i = 0
    w = 0
    for x in range(1,len(predictedVals)+1):
    #get predictions of real price by adding to last day we have iteratively (prevDay:
        if i == 0:
            #This says our first actual price equals the previous price + change in price magnitude
            #predictedPrice_array[i] = prevDay_array[i] + predictedVals[0]
            predictedPrice_array[i] = prevDay_array2 + predictedVals[0]
            errorArray[i] = ((trueVals[i]- predictedPrice_array[i])/trueVals[i])*100

        else:
            #This says we keep adding price magnitude to the next day
            predictedPrice_array[i] = predictedPrice_array[i-1] + predictedVals[i]
            errorArray[i] = ((trueVals[i] - predictedPrice_array[i]) / trueVals[i]) * 100


        #calculating RMSE:
        #rmse_array[m] = sqrt(mean_squared_error(trueVals, predictedPrice_array))
        i += 1

    #reshape our output arrays to be the same dimensions
    predictedPrice_array = predictedPrice_array.reshape(forecast_out,1)
    errorArray = errorArray.reshape(forecast_out,1)
    trueVals = trueVals.reshape(forecast_out,1)
    #put all of our arrays into one numpy array
    allVals = np.hstack((trueVals, predictedPrice_array,errorArray))
    #put all of these values into a dataframe to be exported to CSV file
    df_Output = pd.DataFrame(data= allVals)
    df_Output.columns = ['True Price', 'Predicted Price','Error']
    df_Output.to_csv('Output.csv', sep=',')

    w += 1

    del model



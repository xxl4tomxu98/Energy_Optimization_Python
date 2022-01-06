#import necessary libraries
import numpy as np
import pandas as pd
from pandas import read_csv
from glob import glob
import matplotlib.pyplot as plt


#################################################################
#INPUT PARAMETERS
dischargePowerCap = float(100) #max discharge power capacity in kW
chargePowerCap = float(100) #max charge power capacity in kW
dischargeEnergyCap = float(200) #max discharge energy capacity in kWh
RTefficiency = 0.85 #AC-AC roundtrip efficiency
maxDailyThroughput = float(200) #max daily discharge throughput in kWh
chargeTime = int(dischargeEnergyCap/chargePowerCap) #hour
dischargeTime = int(dischargeEnergyCap/dischargePowerCap) #hr

#INPUTS: names of csv files, name of zone being examined
#read in all csv files at once from directory
#feed in your filenames here for HOURLY LBMP
filenames = glob('2017*.csv')
#get list of dataframes for each day
dataframes= [pd.read_csv(f, header= 0, index_col=0) for f in filenames]
#input ZONE name here
zone = 'N.Y.C.'


#############################################################
#initializing variables:
storedEnergy = 0
dischargedEnergy = 0
dfName_list = []
b = 0
q = 1
for x in range(1,len(dataframes)+1):
     # specify the desired zone
    df1 = dataframes[b]
    df1 = df1[df1.Name == zone]
    table = df1.sort_values(by='LBMP ($/MWHr)', ascending=True)
    times = table.index.values
    # add columns to dataframe containing desired output data
    df1 = df1.assign(Battery_Power_Output=np.nan, SOE=np.nan, Hourly_Revenue_Generation=np.nan,
                     Hourly_Charging_Cost=np.nan, Throughput=np.nan, IntIndex=np.nan)

    # initialize arrays of data that will be inserted into our dataframe
    SOE_array = np.zeros(len(table))
    Battery_Output_array = np.zeros(len(table))
    Hourly_Revenue_array = np.zeros(len(table))
    Hourly_Charging_Cost = np.zeros(len(table))
    Throughput_array = np.zeros(len(table))
    # scale our LBMP values to kW:
    LBMPvals = -df1['Marginal Cost Congestion ($/MWHr)'].values
    # put in a numerical index
    l = [i for i in range(24)]
    df1['IntIndex'] = l

    # sort our given data by price (from lowest to highest price)
    table = df1.sort_values(by='LBMP ($/MWHr)', ascending=True)

    # get our sorted time values into an array
    times = table.index.values

    # get the integer indexes for highest and lowest prices
    chargeIndex = table.iloc[0:2]
    chargeIndex = chargeIndex['IntIndex'].sort_values(ascending=True)
    dischargeIndex = table.iloc[-2:]
    dischargeRef = table.iloc[-1]
    dischargeRef = dischargeRef['IntIndex']#create a reference list ordered by price value to keep track of order by price
    dischargeIndex = dischargeIndex['IntIndex'].sort_values(ascending=True)

    num = len(table)
    # create a loop to go through each hour of the day

    w = 0
    y = 1

    # create an iterator for charge index
    r = 0
    # create an iterator of discharge index
    k = 0
    for z in range(1, len(LBMPvals)):
        print('Z:', z)

        # create an integer index for each day we iterate through
        intIndex = y
        print('IntIndex: ', intIndex)
        print('ChargeIndex: ', chargeIndex[r])
        print('dischargeIndex: ', dischargeIndex[k])

        # conditions where we can charge the battery
        #charge the battery if it is one of the cheapest times and the battery is not already charged:
        if ((intIndex == chargeIndex[r]) & (SOE_array[y-1]<dischargeEnergyCap)):
            # if SOE_array[y-1]<dischargeEnergyCap:
            SOE_array[y] = SOE_array[y - 1] + (chargeTime * chargePowerCap)
            Battery_Output_array[y] = 0  # we do not discharge while charging
            Hourly_Revenue_array[y] = 0  # no revenue is generated while charging
            chargeEnergy = chargePowerCap
            Hourly_Charging_Cost[y] = LBMPvals[y] * chargeTime * 0.001 *chargeEnergy
            Throughput_array[y] = 0  # no discharge throughput while charging
            if r == 0:
                r += 1
            print('CHARGE')
            print(SOE_array[y], Battery_Output_array[y], Hourly_Revenue_array[y], Hourly_Charging_Cost[y],
                  Throughput_array[y])
        # conditions where we discharge the battery
        #charge the battery if the time if LBMP high, the battery has charge, and we have not yet reached our max daily discharging throughput
        elif ((intIndex == dischargeIndex[k]) & (SOE_array[y-1]>0) & (Throughput_array[y]<maxDailyThroughput)):

            if (intIndex == dischargeRef):
                dischargeEnergy = dischargePowerCap
                loss = 0
            else:
                val = dischargePowerCap * RTefficiency #take into account roundtrip efficiency (get less output power than input)
                dischargeEnergy = val
                loss = dischargePowerCap * (1-RTefficiency) #losses that we experience when discharging battery

            SOE_array[y] = SOE_array[y - 1] - dischargeEnergy - loss
            Battery_Output_array[y] = dischargeEnergy
            Hourly_Revenue_array[y] = LBMPvals[y] * dischargeTime *0.001 *dischargeEnergy
            Hourly_Charging_Cost[y] = 0  # no cost for charging while batterry is outputting
            Throughput_array[y] = dischargeEnergy

            if k == 0:
                k += 1
            print('DISCHARGE')
            print(SOE_array[y], Battery_Output_array[y], Hourly_Revenue_array[y], Hourly_Charging_Cost[y],
                  Throughput_array[y])
        # conditions when we are not charging or discharging (based on LBMP price)
        #so the conditions remain the same as the previous value in the array
        else:
            prevVal = SOE_array[y - 1]
            SOE_array[y] = prevVal
            Battery_Output_array[y] = 0
            Hourly_Revenue_array[y] = 0
            Hourly_Charging_Cost[y] = 0
            Throughput_array[y] = 0  # no discharge throughput while charging
            print('ELSE')
            print(SOE_array[y], Battery_Output_array[y], Hourly_Revenue_array[y], Hourly_Charging_Cost[y],
                  Throughput_array[y])

        print('#####################################################')

        w += 1
        y += 1

    df1['Battery_Power_Output'] = Battery_Output_array
    df1['SOE'] = SOE_array
    df1['Hourly_Revenue_Generation'] = Hourly_Revenue_array
    df1['Hourly_Charging_Cost'] = Hourly_Charging_Cost
    df1['Throughput'] = Throughput_array
    dfName = 'new' + filenames[b]
    dfName_list.append(dfName)
    df1.to_csv(dfName, sep= ',')
    b += 1
    q += 1



#read back in our csv files for further calculations
dayFiles = glob('new*.csv')
days= [pd.read_csv(f, header= 0, index_col=0) for f in dayFiles]
#put all of daily hourly data into a single dataframe
df_Year = pd.concat(days)

#Revenues calculations:
annualRevenue = df_Year['Hourly_Revenue_Generation'].sum() #annual Revenue generation
annual_ChargingCost = df_Year['Hourly_Charging_Cost'].sum() #annual charging costs
#Annual throughput:
annualThroughput = df_Year['Throughput'].sum() #annual discharged throughput

##########################################################
#preparing data to be plotted
#split our data up into different weeks
week = 7
hours = 24
weeksList = []

for g, df in df_Year.groupby(np.arange(len(df_Year['Hourly_Revenue_Generation']))//(week*hours)):
    weeksList.append(df)

weeklyProfit = np.zeros(len(weeksList))
k = 0
for x in range(0,len(weeksList)):
    week = weeksList[k]
    weeklyRevenue = week['Hourly_Revenue_Generation'].sum()
    weeklyLoss = week['Hourly_Charging_Cost'].sum()
    weeklyProfit[k] = weeklyRevenue + weeklyLoss
    k += 1

#find the index of the week that yields the max profit
maxProfit_Week= np.argmax(weeklyProfit)

#find the most profitable week:
mostProfitable_week = weeksList[maxProfit_Week]
hourlyLBMP = mostProfitable_week['LBMP ($/MWHr)']
hourly_BatteryDischarge = mostProfitable_week['Throughput']

#PLOT NUMBER ONE
#plot our outputs
# Two subplots, unpack the axes array immediately
plt.rcParams.update({'font.size': 12})
f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(hourlyLBMP)
ax1.set_title('Hourly LBMP for Most Profitable Week')
ax1.set_ylabel('LBMP ($/MWhr)')
ax2.plot(hourly_BatteryDischarge)
ax2.set_title('Hourly Battery Discharge for Most Profitable Week')
ax2.set_ylabel('Battery Discharge (kWhr)')
plt.xticks(rotation=90)
plt.show()



##########################################
#dataframe that contains the hourly revenue generation for all days
df2 = df_Year
#initialize a list that will hold dataframes of all revenue data for each month
monthsList = []
monthNames = []
#take length (number of rows) of input dataframe for if statements
l = len(df_Year['Hourly_Revenue_Generation'])


if l>=(30*24):
    dfJan = df2.iloc[0:(30*24)]
    dfJan_cost = df2.iloc[0:(30 * 24)]
    monthsList.append(dfJan)
    monthNames.append('Jan.')
if l>=(56*24):
    dfFeb = df2.iloc[(30*24):(58*24)]
    monthsList.append(dfFeb)
    monthNames.append('Feb.')
if l>=(90*24):
    dfMar = df2.iloc[(24*58):(89*24)]
    monthsList.append(dfMar)
    monthNames.append('Mar.')
if l>=(119*24):
    dfApr = df2.iloc[(89*24):(119*24)]
    monthsList.append(dfApr)
    monthNames.append('Apr.')
if l>=(150*24):
    dfMay = df2.iloc[(119*24):(150*24)]
    monthsList.append(dfMay)
    monthNames.append('May')
if l>=(180*24):
    dfJune = df2.iloc[(150*24):(180*24)]
    monthsList.append(dfJune)
    monthNames.append('June')
if l>= (211*24):
    dfJuly = df2.iloc[(180*24):(211*24)]
    monthsList.append(dfJuly)
    monthNames.append('July')
if l>=(242*24):
    dfAug = df2.iloc[(211*24):(242*24)]
    monthsList.append(dfAug)
    monthNames.append('Aug.')
if l>= (272*24):
    dfSept = df2.iloc[(242*24):(272*24)]
    monthsList.append(dfSept)
    monthNames.append('Sept.')
if l>= (303*24):
    dfOct = df2.iloc[(272*24):(303*24)]
    monthsList.append(dfOct)
    monthNames.append('Oct.')
if l>= (334*24):
    dfNov = df2.iloc[(303*24):(333*24)]
    monthsList.append(dfNov)
    monthNames.append('Nov.')
if l>= (364*24):
    dfDec = df2.iloc[(333*24):(364*24)]
    monthsList.append(dfDec)
    monthNames.append('Dec.')

#initialize an array that will contain the monthly revenue values for each month
monthlyProfit = np.zeros(len(monthsList))
k = 0 #initialize iterator for loop

#Calculat the months revenues, charging costs, and profits
for x in range(0,len(monthsList)):
    month = monthsList[k]
    monthlyRevenue = month['Hourly_Revenue_Generation'].sum()
    monthlyLoss = month ['Hourly_Charging_Cost'].sum()
    monthlyProfit[k] = monthlyRevenue - monthlyLoss
    k += 1

#PLOT NUMBER 2:
#plot total profit for each month:
plt.plot(monthNames,monthlyProfit)
plt.ylabel('Monthly Profit($)')
plt.show()

#OUTPUTS:
#baattery storage:
dfPowerOutput = pd.DataFrame(data= df_Year['Throughput'])
dfSOE = pd.DataFrame(data= df_Year['SOE'])
batteryStorage_Output = pd.concat([dfPowerOutput, dfSOE], axis= 1)
batteryStorage_Output.columns = ['Power Output (kW)', 'State of Energy (kWhr)']
batteryStorage_Output.to_csv('batteryStorage_Output.csv', sep =',')
#revenue (for storage system):
totalAnnualRevenue = df_Year['Hourly_Revenue_Generation'].sum()
totalAnnualChargingCost = df_Year['Hourly_Charging_Cost'].sum()
totalAnnualDischarged_Throughput = df_Year['Throughput'].sum()
#df_RevenueOutput = pd.concat([totalAnnualRevenue, totalAnnualChargingCost, totalAnnualDischarged_Throughput], axis= 1)
RevenueOutput = np.array([totalAnnualRevenue, totalAnnualChargingCost, totalAnnualDischarged_Throughput])
RevenueOutput = RevenueOutput.reshape(1,3)
df_RevenueOutput = pd.DataFrame(data= RevenueOutput)
df_RevenueOutput.columns = ['Total Annual Revenue ($)', 'Total Annual Charging Cost ($)', 'Total Annual Discharged Throughput (kWhr)']
df_RevenueOutput.to_csv('Revenue_Output.csv', sep= ',')




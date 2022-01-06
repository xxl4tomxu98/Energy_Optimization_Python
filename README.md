# Power-Modeling-Tool
Purpose: Analysis of revenue generation from battery storage system that is performing energy arbitrage by participating in NYISO day-ahead energy market. 

Please note that the final system design is included in the file "FinalSystem.py".
Inputs:
  1. Battery Storage Design 
  
  -Max power capacity (both charge and discharge in kW)
  
  -Discharge energy capacity (in kWh)
  
  -AC-AC Roundtrip efficiency 
  
  -Max daily discharged throughput (in kWh)
  
  2. Market Price inputs:
  
  -Hourly LMBPS (write the beginning names of the csv files, i.e., the year for the variable named 'filenames')
  (Please note that examples of input CSV files of historical NYISO data can be found within the referrals of this repository) 
  
  -Zone (any in NY state) 
  
 Outputs:
 
  -CSV file with power output (kW) & State of Energy (kWh)
  
  -CSV file with total annual revenue generation, total annual charging cost, and total annual throughput 
  
 NOTE:
 I've added an extra file called 'dayAheadPrediction.' I've created a neural network to assist the project with energy price projections, but is in its early stages and requires more development, tuning and data to become effective. I figured it couldn't hurt to attempt to solve the problem of next-day forecasts, so I included it here! 
 
 
 EXTRA NOTES ON THE SYSTEM: 
 
 Overall System Requirements:
 
1. The system SHALL optimize the battery storage dispatch (with an optimization time horizon of at
least 1 day) for the day ahead energy market

○ The battery storage’s State of Energy SHALL be continuous between optimization time
horizon boundaries

2. The system SHALL accept the following as inputs for the battery storage asset:

○ Max discharge power capacity (kW)

○ Max charge power capacity (kW)

○ Discharge energy capacity (kWh)

○ AC-AC Round-trip efficiency (%)

○ Maximum daily discharged throughput (kWh)

3. The system SHALL accept the following as inputs for the market revenues:

○ Hourly LBMP ($/MWh)

○ Zone

4. The system SHALL output the following values about a given battery storage system, for a year’s
worth of data, at an hourly resolution

○ Power output (kW)

○ State of Energy (kWh)

5. The system SHALL output the following summary values about a given storage system:

○ Total annual revenue generation ($)

○ Total annual charging cost ($)

○ Total annual discharged throughput (kWh)

6. The system SHALL output the following plots

○ A plot that includes both hourly battery dispatch and hourly LBMP for the most
profitable week

○ A plot that shows the total profit for each month

System Requirements Used:

Model that meets the Overall System Requirements and uses the following
inputs and assumptions:

1. Battery storage design inputs:

■ Max power capacity (both charge and discharge) = 100 kW

■ Discharge energy capacity = 200 kWh

■ AC-AC round trip efficiency = 85%

■ Maximum daily discharged throughput (kWh) = 200 kWh

2.Market prices inputs:

■ 2017 Hourly LBMPs (examples of these CSVs can be found in the releases of this repository) 

■ Zone = N.Y.C.

3. Assumptions

■ The battery storage system has 100% depth of discharge capabilities

■ The battery storage system does not experience any degradation during the first
year of operation

■ The battery storage system is a price taker (i.e. receives the LBMP as the market
price)

■ The battery storage system charging cost and discharging revenue should both
be based on the wholesale LBMPs

■ The historical LBMP data can be used directly as a proxy for price forecasts (i.e.
no need to forecast future energy prices




  

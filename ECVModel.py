import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

#First file for creating the equivalent series model

#Loading in datqa for use with the functions
#load in the data for the battery discharge (and temperature)
#Ensure that you have put the correct file location in the string here!
df = pd.read_csv(r"C:\Users\thoma\OneDrive\Desktop\FCCTPython\FCTT 18-19 Data\Battery_Testing_Data.csv")
t = np.array(df['Time (s)'])[1:]
i = np.array(df['Current (mA)'])[1:]/1000
V = np.array(df['Voltage (V)'])[1:]
Temp = np.array(df['Temperature'])[1:]

#Importing the SOC/OCV data
df2 = pd.read_csv(r"C:\Users\thoma\OneDrive\Desktop\FCCTPython\FCTT 18-19 Data\SOC_OCV_MFCTT_2019.csv", delimiter=('\t'))
SOCDataSheet = np.array(df2['SOC'])/100
SOCDataSheetV = np.array(df2['Ecell/V'])

#Battery data
iResistance = 0.01624 #Estimated from the datasheet
batteryCapacity = 2.5


#This model first requires the open circuit voltage. ------------------
#This state of charge function should take the current time history
#and gives a state of charge
def SOCSimple(current, time, capacity, SOC = 0.999999998, eta = 0.99):
    z = np.zeros(np.shape(time))
    z[0] = SOC

    #Using a for loop to differentiate between charging and not
    for j in range(1,len(t)):

        #Assumes discharging efficiency = 1, otherwise no change
        if current[j] < 0:
            eta = 1

        #Calculation done here
        z[j] = z[j-1] + eta * (time[j] - time[j-1]) * current[j] / (capacity * 3600)

    #Return SOC as an array
    return z



#This function will recieve an SOC value, and return a OCV ----------------
def SOCtoOCV(SOC, SOCDataSheet, SOCDataSheetV):
    #This process is simplified as the provided data is constantly increasing

    #Check the data is in range so it doesnt break the function
    if SOC < np.min(SOCDataSheet) or SOC > np.max(SOCDataSheet):
        sys.exit('Provided SOC is outside the range of the datasheet')

    for j in range(0,len(SOCDataSheet)):
        #Locate the value of SOC
        if SOC > SOCDataSheet[j]:
            #Return a Linearly interpolated value
            return SOCDataSheetV[j] - (SOCDataSheetV[j-1] - SOCDataSheetV[j]) \
                * (SOCDataSheet[j-1] - SOC)/(SOCDataSheet[j-1] - SOCDataSheet[j])




#This following funciton will take an OCV for the cell, a current,
#and an internal resistance to give a voltage at the output
def theveninVoltage(OCV, current, iResistance):
    return OCV + current * iResistance




#This function should take an array of the current, and provide a voltage
#Using the previously defined functions
def theveninModel(current, t, iResistance, SOCDataSheet, SOCDataSheetV, batCap, initialSOC):
    #This function will first get the simhistory for SOC (z)
    z = SOCSimple(current, t, batCap, SOC = initialSOC)

    #Will now take the SOC history and get a (open circuit) voltage history
    OCV = np.zeros(np.shape(t))
    for j in range(0, len(t)):
        OCV[j] = SOCtoOCV(z[j], SOCDataSheet, SOCDataSheetV)

    #Take the OCV history and return a thevenin equivalent voltage
    V = theveninVoltage(OCV, current, iResistance)

    return V, z


#
# predictedVoltage, z = theveninModel(i, t, iResistance, SOCDataSheet, SOCDataSheetV, batteryCapacity, 0.85)
#
# z = SOCSimple(i, t, batteryCapacity, SOC = 0.85, eta = 0.99)
#
# start = 0
# end = 10000
# plt.plot(t[start:], predictedVoltage[start:])
# plt.plot(t[start:], V[start:])
# # plt.ylim([4, 4.25])
# plt.show()
#
# print(predictedVoltage[0])
# print(V[0])

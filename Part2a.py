import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ECVModel import *
import scipy.interpolate as sci
import Part2Functions as f2

#First, just getting the data out of the csv file
t20, i20, V20 = f2.csvOpener(r"C:\Users\thoma\OneDrive\Desktop\FCCTPython\FCTT 18-19 Data\Model_Training_Data_20.csv")

"""
Part 2.a.a
Getting all of the R0, C1, R1
"""
#The states of charge for the pulses are known;
socs20 = np.linspace(99,20,72)
socs20 -= socs20%10

#Calling the initial trainer function on the 20 degree C data
parameters_20deg = f2.batteryTrainerInitial(t20, i20, V20, socs20)
#This gives data in the following order;
#R0, R1, C1, dV, i, state of charge, current
#0   1   2   3   4      5               6

"""
Part 2.a.c-e
Getting the averages and putting them back into the model
"""
#Taking the average to put back into the model
R0_20deg = np.mean(parameters_20deg[:,0])
C1 = np.mean(parameters_20deg[2])

rev_parameters_20deg, R_20deg_60SOC = f2.revisedTrainer(parameters_20deg, R0_20deg, C1)

for i in parameters_20deg:
    if i[4] == -2.5:
        plt.scatter(i[5],i[0])
plt.show()


"""
Part 2.a.f
Plotting as required
"""

plot_SOC = 60
plotIndices = []
for j in range(0, len(rev_parameters_20deg)):
    if rev_parameters_20deg[j,5] == plot_SOC:
        plotIndices.append(j)

# plt.scatter(rev_parameters_20deg[plotIndices,4], rev_parameters_20deg[plotIndices,1])
# plt.show()

"""
Part 2.a.g
fitting the gaussian(ish)
"""
fitX = np.array(rev_parameters_20deg[plotIndices,4])
fitY = np.array(rev_parameters_20deg[plotIndices,1])
sortIndex = np.argsort(fitX)
fitX = fitX[sortIndex]
fitY = fitY[sortIndex]

#Averaging the two pulses where its 4
fitX[-2] = np.mean(fitX[-2:])
fitY[-2] = np.mean(fitY[-2:])
fitX = fitX[:-1]
fitY = fitY[:-1]
# plt.scatter(fitX,fitY)
#Fitting the (almost) gaussian distribution and plotting
coefs_20deg_60SOC = f2.gaussFitter(fitX, fitY)
R10A_20deg_60SOC = fitY[np.argmin(np.abs(fitX))]
fineI = np.linspace(-20,5,200)
fineR1_20dog_60SOC = f2.modGaussa(fineI, R10A_20deg_60SOC, coefs_20deg_60SOC)
plt.plot(fineI, fineR1_20dog_60SOC)
plt.show()

# print(coefs_20deg_60SOC)
# print(R10A_20deg_60SOC)

#OLD CODE KEPT FOR REFERENCE BELOW --------------------------------------------
#------------------------------------------------------------------------------

# #To locate the charging and discharging, we can use the times when i != 0
# #This function will recieve the arrays, and give a start time, end time, and
# #current maginitude.
# def pulseLocator(iIn, tIn):
#     startTimes = []
#     startIndex = []
#     endTimes = []
#     endIndex = []
#     iMags = []
#     trigger = 0

#     for j in range(0,len(tIn)):
#         if iIn[j] != 0:
#             if trigger == 0:
#                 startTimes.append(tIn[j])
#                 startIndex.append(j)
#                 iMags.append(iIn[j+1])
#                 trigger += 1
#         else:
#             if trigger > 0:
#                 endTimes.append(tIn[j-1])
#                 endIndex.append(j-1)
#                 trigger = 0


#      #This function returns data about each pulse, where to get info about the
#     #nth pulse, look for array[:,n]
#     return np.array(([startTimes, endTimes, iMags, startIndex, endIndex]))


# #This function will recieve the rest of the time history of the training data
# #to valcultae R0 R1 and C1. First timestep it revcieves should be last with
# #the current being driven trhough it
# def impedances(tIn, VIn, iIn):
#     #R0 from inital voltage drop
#     R0i = (VIn[0] - VIn[1]) / iIn

#     #This part of the code is to find the relaxation time
#     #Expects relaxation after 3000 t steps
#     j = 0
#     while abs(VIn[j] - VIn[-1]) > 0.00001:
#         j+=1

#     dV = (VIn[0] - VIn[j])

#     #Calculating R1 with R0
#     R1 = (dV / iIn) - R0i

#     #Calculating C1
#     C1 = (tIn[j] - tIn[0]) / (4 * R1)

#     return [R0i, R1, C1, dV, iIn]


# #This function will generate the arrays of battery data as required, using the
# #pulse locator funciton as required
# def batteryTrainerInitial(tIn, iIn, VIn, socsIn):

#     #Using the funciton to find the data about pulses:
#     pulseData = pulseLocator(iIn, tIn)

#     #Will go through all of the pulses to do calculations on
#     datapoints = []

#     for j in range(0, len(pulseData[0,:])):
#         #We will pass the data from the following index on to the function
#         calcIndex = int(pulseData[4,j])

#         #We need when the next pulse starts so we can get the steady state value
#         if j != len(pulseData[0,:]) - 1:
#             endIndex = int(pulseData[3,j+1]) - 1
#         else:
#             endIndex = len(tIn) - 1

#         currentPulse = impedances(tIn[calcIndex:endIndex], VIn[calcIndex:endIndex], pulseData[2,j])
#         currentPulse.append(socsIn[j])
#         currentPulse.append(pulseData[2,j])
#         datapoints.append(currentPulse)

#     #This function will return the information in the following order:
#     #R0, R1, C1, dV, i, state of charge, current
#     return np.array(datapoints)



# #Just generating an array for the various states of charge
# socs = np.full(9 * 8, 20)
# for j in range(0 , 8):
#     for k in range(9 * j, 9 * j + 9):
#         socs[k] += 10 * j
# socs = np.flip(socs)



# #Calling the above functions to return the array of battery parameters
# initialData = batteryTrainerInitial(t, i, V, socs)

# plt.(initialData[:,0],initialData[:,5])
# plt.show()


# """
# For 2a1e
# """

# #Will now use this data to plug it into a new functiion, based on the previous
# #Will also aim to ouput the data in a format which is useful to us
# def revisedTrainer(iData):
#     R0i = np.mean(iData[:,0])
#     C1i = np.mean(iData[:,2])
#     # print(C1)
#     R60 = np.mean(iData[17:27,0])
#     # print(R60)

#     R1s = np.zeros(len(iData[:,0]))
#     #First we will find all our values of R1, we can sort them after
#     for j in range(0, len(iData)):
#         R1s[j] = (iData[j,3] / iData[j,4]) - R0i


#     # The currents and SOCs to be used for the arrays
#     currentsi = np.round(iData[:9,4], 2)
#     socsi = np.arange(20,100,10)

#     #Creating the matrix of R1 values
#     R1Mi = np.zeros((len(currentsi),len(socsi)))
#     for j in range(0,len(socsi)):
#         R1Mi[:,j] = R1s[j*9:j*9+9]


#     #This section sorts the R1s according to the currents
#     sortingIndex = np.argsort(currentsi)
#     R1Msorted = np.zeros(np.shape(R1Mi))
#     R1Msorted[:,:] = R1Mi[sortingIndex,:]
#     currentSorted = currentsi[sortingIndex]



#     #CURRENTLY UP TO HERE ON SORTING OUT MAKING THE THING THE AVERAGE OF THE 4S
#     #As the current 4 is used twice, will average out the last two terms
#     R1Msorted[-2,:] = (R1Msorted[-1,:] + R1Msorted[-1,:]) / 2
#     return R1Msorted[:-1,:], R0i, C1i, currentSorted[:-1], socsi, R60




# # start = 140000
# # end = 280000
# # plt.plot(t[start:end],i[start:end])
# # plt.show()
# # plt.plot(t[start:end],V[start:end])
# # plt.show()

# R1M, R0, C1, tableCurrents, tableSocs, R2060 = revisedTrainer(initialData)


# #We need a higher resolution of the data, so will use this spline function
# #from scipy which works in two dimensions
# f = sci.interp2d(tableCurrents, tableSocs, R1M, kind = 'cubic')
# currentFine = np.linspace(-20,4,100)
# SOCfine = np.linspace(20,90,100)
# interpR1 = f(currentFine, SOCfine)

# # #Can show this data if desired:
# # plt.contourf(tableCurrents,tableSocs,R1M)
# # plt.show()
# # plt.contourf(currentFine, SOCfine, interpR1)
# # plt.colorbar()
# # plt.show()


# """
# Part 2a1f
# """

# #Plotting the values of R1 as current values with contant soc
# # plt.plot(tableCurrents,R1M[:,2])
# # plt.xlabel('Current /A')
# # plt.ylabel('R1 / ohms')


# """
# Part 2a1g
# """

# #Next up, going to fit the gaussian
# from scipy.optimize import curve_fit


# #Define a function which recieves the current and resistances
# def gaussFitter(iIn, R1In):


#     #Adding the smallest current resistnace again as the 0 value
#     R10A = R1In[4]
#     iIn = np.concatenate((iIn[:5],np.array([0]),iIn[5:]))
#     R1In = np.concatenate((R1In[:5],np.array([R10A]),R1In[5:]))

#     #Define our gaussian function
#     def gauss(iloc, bloc, cloc):
#         y = R10A * np.exp(-(iloc - bloc)**2 / cloc)
#         return y

#     #Using the fitting function from scipy
#     parameters, covariance =  curve_fit(gauss, iIn, R1In)
#     xfine = np.linspace(-20,5,200)
#     yfine = gauss(xfine, parameters[0], parameters[1])
#     plt.plot(iIn, R1In)
#     plt.plot(xfine,yfine)


# gaussFitter(tableCurrents, R1M[:,2])

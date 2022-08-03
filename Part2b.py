import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Part2a import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import Part2Functions as f2
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

"""
Part a,
First getting the data and doing the inital analysis
"""
t0, i0, V0 = f2.csvOpener(r"C:\Users\thoma\OneDrive\Desktop\FCCTPython\FCTT 18-19 Data\Model_Training_Data_0.csv")
t20, i20, V20 = f2.csvOpener(r"C:\Users\thoma\OneDrive\Desktop\FCCTPython\FCTT 18-19 Data\Model_Training_Data_20.csv")
t40, i40, V40 = f2.csvOpener(r"C:\Users\thoma\OneDrive\Desktop\FCCTPython\FCTT 18-19 Data\Model_Training_Data_40.csv")

socs0 = np.linspace(99,20,72)[:-2]
socs0 -= socs0%10

socs20 = np.linspace(99,20,72)
socs20 -= socs20%10

socs40 = np.linspace(99,20,72)
socs40 -= socs40%10

# Getting the value of C1 from part 2a
C1 = 3110.782132503249


#Getting the data out in the following form:
#R0, R1, C1, dV, i, state of charge, current
#0   1    2   3  4      5              6
parameters_0deg = f2.TrainerFixedC1(t0, i0, V0, socs0, C1)
R0_0deg = np.mean(parameters_0deg[:,0])
rev_parameters_0deg, R_0deg_60SOC = f2.revisedTrainer(parameters_0deg, R0_0deg, C1)

parameters_20deg = f2.TrainerFixedC1(t20, i20, V20, socs20, C1)
R0_20deg = np.mean(parameters_20deg[:,0])
rev_parameters_20deg, R_20deg_60SOC = f2.revisedTrainer(parameters_20deg, R0_20deg, C1)
# print(parameters_20deg[:,1])


parameters_40deg = f2.TrainerFixedC1(t40, i40, V40, socs40, C1)
R0_40deg = np.mean(parameters_40deg[:,0])
rev_parameters_40deg, R_40deg_60SOC = f2.revisedTrainer(parameters_40deg, R0_40deg, C1)


"""
Next up is to fit the correct arrays to their own gaussian functions (i think)
doing the 0 deg one first
"""

plot_SOC = 60
plotIndices_0deg = []
for j in range(0, len(rev_parameters_0deg)):
    if rev_parameters_0deg[j,5] == plot_SOC:
        plotIndices_0deg.append(j)


currentFit, R1_60SOC_0deg = f2.pulseCleaner(rev_parameters_0deg[plotIndices_0deg,4],\
                                            rev_parameters_0deg[plotIndices_0deg,1])
plt.scatter(currentFit, R1_60SOC_0deg)
coefs_0deg_60SOC = f2.gaussFitter(currentFit, R1_60SOC_0deg)
R10A_0deg_60SOC = R1_60SOC_0deg[np.argmin(np.abs(currentFit))]

fineI = np.linspace(-20,5,200)
fineR1_0deg_60SOC = f2.modGaussa(fineI, R10A_0deg_60SOC, coefs_0deg_60SOC)


"""
Big batch of plotting code below.
"""
#
# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
# fig.set_facecolor('white')
#
# ax.plot(fineI, fineR1_0deg_60SOC,linestyle = '--',c='cornflowerblue',zorder=10, clip_on=False, label = r'Modified Gaussian function')
# ax.scatter(rev_parameters_0deg[plotIndices,4], rev_parameters_0deg[plotIndices,1],c='r',marker='o',zorder=10, clip_on=False,label = r"$R_1$ for 0$^{\circ}$C and 60\% $SOC$")
#
# ax.set_xlabel(r"$i(t)$ [A]",fontsize=14)
# ax.set_ylabel(r"$R_1$ [$\Omega$]", color="k", fontsize=14)
#
#
# """""
# Gridlines & ticks
# """""
# ax.minorticks_on()
# ax.grid(which='major', linestyle='-', linewidth='0.5',color='silver')
# ax.grid(which='minor', linestyle=':', linewidth='0.5',color='silver')
#
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# """""
# Framing
# """""
# ax.set_ylim([0,0.12])
# ax.set_xlim([-21,5])
#
# """""
# Legend
# """""
# handles,labels = ax.get_legend_handles_labels()
#
# handles = [handles[1], handles[0]]
# labels = [labels[1], labels[0]]
# ax.legend(handles,labels,handlelength=1.5,shadow=True,fontsize=12,loc='upper left')
#
# """""
# saving high res
# """""
# plt.show()
# plt.savefig('plot2a.pdf',dpi=300, bbox_inches = "tight")
# #

"""
Repeat of the 20 degrees one
"""
# plot_SOC = 60
# plotIndices_20deg = []
# for j in range(0, len(rev_parameters_20deg)):
#     if rev_parameters_20deg[j,5] == plot_SOC:
#         plotIndices_20deg.append(j)

# currentFit, R1_60SOC_20deg = f2.pulseCleaner(rev_parameters_20deg[plotIndices_20deg,4],\
#                                    rev_parameters_20deg[plotIndices_20deg,1])


# plt.scatter(currentFit, R1_60SOC_20deg)
# coefs_20deg_60SOC = f2.gaussFitter(currentFit, R1_60SOC_20deg)
# R10A_20deg_60SOC = R1_60SOC_20deg[np.argmin(np.abs(currentFit))]

# fineI = np.linspace(-20,5,200)
# fineR1_20deg_60SOC = f2.modGauss(fineI, R10A_20deg_60SOC, coefs_20deg_60SOC)
# plt.plot(fineI, fineR1_20deg_60SOC)
# plt.show()

"""
And finaly the 40 degrees one
"""

plot_SOC = 60
plotIndices_40deg = []
for j in range(0, len(rev_parameters_40deg)):
    if rev_parameters_40deg[j,5] == plot_SOC:
        plotIndices_40deg.append(j)


currentFit, R1_60SOC_40deg = f2.pulseCleaner(rev_parameters_40deg[plotIndices_40deg,4],\
                                            rev_parameters_40deg[plotIndices_40deg,1])
plt.scatter(currentFit, R1_60SOC_40deg)
coefs_40deg_60SOC = f2.gaussFitter(currentFit, R1_60SOC_40deg)
R10A_40deg_60SOC = R1_60SOC_40deg[np.argmin(np.abs(currentFit))]

fineI = np.linspace(-20,5,200)
fineR1_40deg_60SOC = f2.modGaussa(fineI, R10A_40deg_60SOC, coefs_40deg_60SOC)
# plt.plot(fineI, fineR1_40deg_60SOC)
# plt.xlabel('Current /A')
# plt.ylabel('R1 /ohms')
# plt.show()

"""
More plotting code below
"""
#
# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
# fig.set_facecolor('white')
#
# ax.plot(fineI, fineR1_40deg_60SOC,linestyle = '--',c='cornflowerblue',zorder=10, clip_on=False, label = r'Modified Gaussian function')
# ax.scatter(rev_parameters_40deg[plotIndices,4], rev_parameters_40deg[plotIndices,1],c='r',marker='o',zorder=10, clip_on=False,label = r"$R_1$ for 40$^{\circ}$C and 60\% $SOC$")
#
# ax.set_xlabel(r"$i(t)$ [A]",fontsize=14)
# ax.set_ylabel(r"$R_1$ [$\Omega$]", color="k", fontsize=14)
#
#
# """""
# Gridlines & ticks
# """""
# ax.minorticks_on()
# ax.grid(which='major', linestyle='-', linewidth='0.5',color='silver')
# ax.grid(which='minor', linestyle=':', linewidth='0.5',color='silver')
#
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# """""
# Framing
# """""
# ax.set_ylim([0,0.12])
# ax.set_xlim([-21,5])
#
# """""
# Legend
# """""
# handles,labels = ax.get_legend_handles_labels()
#
# handles = [handles[1], handles[0]]
# labels = [labels[1], labels[0]]
# ax.legend(handles,labels,handlelength=1.5,shadow=True,fontsize=12,loc='upper left')
#
# """""
# saving high res
# """""
# plt.show()
# plt.savefig('plot2a.pdf',dpi=300, bbox_inches = "tight")


"""
2.b.b - plotting R0 as a function of temperature
"""
#Just creating the two arrays
temps = np.linspace(0,40,3) + 273.15
R0_array = np.array([R0_0deg, R0_20deg, R0_40deg])
# plt.scatter(temps, R0_array)
# plt.show()


"""
2.b.c - fitting R0 to the arrhenius
"""
#Calling the arrfitter function which uses scipy curve_fit
E_R0 = f2.arrfitter(temps, R0_array, 1)
print(E_R0)
tempsfine = np.linspace(270,320,200)
R0_pred = f2.arrhenius(tempsfine, R0_20deg, E_R0, temps[1])
# plt.plot(tempsfine, R0_pred)
# plt.show()

"""
More plotting
"""
#
# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
# fig.set_facecolor('white')
#
# ax.plot(tempsfine, R0_pred,linestyle = '--',c='cornflowerblue',zorder=10, clip_on=False, label = r'Fitted Arrhenius function')
# ax.scatter(temps, R0_array, c='r',marker='o',zorder=10, clip_on=False,label = r"$R_0$ at 60\% $SOC$ averaged across currents")
#
#
# ax.set_xlabel(r"$T$ [$^{\circ} C$]",fontsize=14)
# ax.set_ylabel(r"$R_0$ [$\Omega$]", color="k", fontsize=14)
#
#
# """""
# Gridlines & ticks
# """""
# ax.minorticks_on()
# ax.grid(which='major', linestyle='-', linewidth='0.5',color='silver')
# ax.grid(which='minor', linestyle=':', linewidth='0.5',color='silver')
#
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# """""
# Framing
# """""
# # ax.set_ylim([0,0.12])
# # ax.set_xlim([-21,5])
#
# """""
# Legend
# """""
# handles,labels = ax.get_legend_handles_labels()
#
# handles = [handles[1], handles[0]]
# labels = [labels[1], labels[0]]
# ax.legend(handles,labels,handlelength=1.5,shadow=True,fontsize=12,loc='upper right')
#
# """""
# saving high res
# """""
# # plt.savefig('plot2a.pdf',dpi=300, bbox_inches = "tight")
# plt.show()
# plt.savefig('plot2a.pdf',dpi=300, bbox_inches = "tight")





#OLD CODE KEPT BELOW - LOOK IF YOU DARE! IT'S UGLY!! --------------------------
#------------------------------------------------------------------------------

# SOCplot = 60
# R0_60SOC = []
# T = np.linspace(0,40,3) + 273.15

# R0_60SOC_0deg = []
# for j in datapoints0:
#     if j[5] == SOCplot:
#         R0_60SOC_0deg.append(j[0])

# R0_60SOC_20deg = []
# for j in datapoints20:
#     if j[5] == SOCplot:
#         R0_60SOC_20deg.append(j[0])

# R0_60SOC_40deg = []
# for j in datapoints40:
#     if j[5] == SOCplot:
#         R0_60SOC_40deg.append(j[0])


# R0_60SOC = np.array([np.mean(R0_60SOC_0deg), np.mean(R0_60SOC_20deg), \
#                       np.mean(R0_60SOC_40deg)])
# print('The resistances from training data are ' + str(R0_60SOC))

# plt.scatter(T, R0_60SOC)

# E_R0_60SOC = f2.arrfitter(T, R0_60SOC, 1)

# print('The fitting term is ' + str(E_R0_60SOC))

# Tfine = np.linspace(270,320,200)
# R0_60SOC_fine = f2.arrhenius(Tfine, R0_60SOC[1], E_R0_60SOC, 293.15)
# plt.plot(Tfine, R0_60SOC_fine)
# plt.show()






# Previous way of doing it can be found below. Less clean!!

# #Using an updated function for the 0deg data
# def revisedTrainer0(iData, C1In):

#     R00 = np.mean(iData[:,0])
#     R060 = np.mean(iData[17:27,0])


#     R1s = np.zeros(len(iData[:,0]))
#     #First we will find all our values of R1, we can sort them after
#     for j in range(0, len(iData)):

#         R1s[j] = (iData[j,3] / iData[j,4]) - R00



#     # The currents and SOCs to be used for the arrays
#     currentsi = np.round(iData[:9,4], 2)
#     socsi = np.arange(20,100,10)

#     #Creating the matrix of R1 values
#     R1Mi = np.zeros((len(currentsi),len(socsi)))
#     for j in range(0,len(socsi)-1):
#         R1Mi[:,j] = R1s[j*9:j*9+9]

#     R1Mi[0:7,len(socsi)-1] = R1s[-7:]

#     #This section sorts the R1s according to the currents
#     sortingIndex = np.argsort(currentsi)
#     R1Msorted = np.zeros(np.shape(R1Mi))
#     R1Msorted[:,:] = R1Mi[sortingIndex,:]
#     currentSorted = currentsi[sortingIndex]



#     #Averaging the last 2 columns as they are both at 4 A
#     R1Msorted[-2,:] = (R1Msorted[-1,:] + R1Msorted[-1,:]) / 2

#     return R1Msorted[:-1,:], R00, C1In, currentSorted[:-1], socsi, R060



# #Need a further function, which is different from the original in that it recieves the
# #Value of C1 prior
# def revisedTrainer40(iData,C1In):
#     R0 = np.mean(iData[:,0])
#     R60 = np.mean(iData[17:27,0])

#     R1s = np.zeros(len(iData[:,0]))
#     #First we will find all our values of R1, we can sort them after
#     for j in range(0, len(iData)):

#         R1s[j] = (iData[j,3] / iData[j,4]) - R0



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



#     #Averaging the last 2 columns as they are both at 4 A
#     R1Msorted[-2,:] = (R1Msorted[-1,:] + R1Msorted[-1,:]) / 2

#     return R1Msorted[:-1,:], R0, C1In, currentSorted[:-1], socsi, R60




# #Need to generate the states of charge for this training data
# socs0 = np.full(70,90)
# for j in range(0, len(socs0)):
#     sub = (10/9) * (j - (j%9))
#     socs0[j]-=sub

# #This function will return the information in the following order:
# #R0, R1, C1, dV, i, state of charge, current (reminder!)
# initialData0 = batteryTrainerInitial(t0, i0, V0, socs0)

# #Defining C1 from the 20 deg data
# C1 = 3110.782132503249

# R1M0, R00, C1, tableCurrents0, tableSocs0, R060 = revisedTrainer0(initialData0, C1)
# # plt.scatter(tableCurrents0, R1M0[:,4])




# """
# # Now just doing the section for 40 deg
# """
# socs40 = np.full(9 * 8, 20)
# for j in range(0 , 8):
#     for k in range(9 * j, 9 * j + 9):
#         socs40[k] += 10 * j
# socs40 = np.flip(socs40)



# initialData40 = batteryTrainerInitial(t40, i40, V40, socs40)

# R1M40, R040, C1, tableCurrents40, tableSocs40, R4060 = revisedTrainer40(initialData40,C1)





# # print(R060)
# # print(R2060)
# # print(R4060)
# # plt.plot(t40,i40)

# #Output the R0 values at different temperatures, where SOC = 60%

# R060SOC = np.array([0.03935283306182192, 0.019154008016032117, 0.014876675454462173])
# temps = np.array([0,20,40],dtype = float)

# #Next, we will fit the Arrhenius equation

# plt.scatter(temps+273.15, R060SOC)






# def arrfitter(x, y, T0pos):
#     x+=273.15
#     T0 = x[T0pos]
#     RT0 = y[T0pos]
#     print(x)
#     print(y)
#     def arr(Tf, E):
#         # Tf += 273.15
#         return RT0 * np.exp((-E / 8.3114) * ((1 / Tf) - (1 / T0)))

#     parameters, covariance =  curve_fit(arr, x, y)
#     print(parameters)
#     xfine = np.linspace(270,320,200)
#     yfine = arr(xfine, parameters[0])
#     plt.plot(xfine,yfine)
#     return parameters[0]


# arrfitter(temps, R060SOC, 1)
# plt.show()

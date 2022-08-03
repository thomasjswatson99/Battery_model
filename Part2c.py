import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import Part2Functions as f2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



#Importing the various datas to this script
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

parameters_20deg = f2.TrainerFixedC1(t20, i20, V20, socs20, C1)

parameters_40deg = f2.TrainerFixedC1(t40, i40, V40, socs40, C1)

# parameters_0deg = f2.batteryTrainerInitial(t0, i0, V0, socs0)
#
# parameters_20deg = f2.batteryTrainerInitial(t20, i20, V20, socs20)
#
# parameters_40deg = f2.batteryTrainerInitial(t40, i40, V40, socs40)


"""
Part 2ca - R1 as a function of SOC
"""
#Comment in and out these quotes below to use the sections of code
# """
currentPlot = -2.5
R1_n2_5A = []
SOCS_R1plot = []
for j in parameters_0deg:
    if j[4] == currentPlot:
        R1_n2_5A.append(j[1])
        SOCS_R1plot.append(j[5])

# plt.plot(SOCS_R1plot, R1_n2_5A)
# plt.show()

# """

# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
# fig.set_facecolor('white')
#
# ax.plot(SOCS_R1plot, R1_n2_5A,linestyle = '--',c='cornflowerblue',zorder=10, clip_on=False)
# ax.scatter(SOCS_R1plot, R1_n2_5A, c='r',marker='o',zorder=10, clip_on=False)
#
#
# ax.set_xlabel(r"$SOC$ $\%$",fontsize=14)
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
# saving high res
# """""
# # plt.savefig('plot2a.pdf',dpi=300, bbox_inches = "tight")
# plt.show()

"""This section below was just troubleshooting **************************
data = f2.TrainerFixedC1(t40, i40, V40, socs40, C1)
plotx = []
ploty = []
for i in data:
    if i[4] == -2.5:
        plotx.append(i[5])
        ploty.append(i[1])
plt.scatter(plotx,ploty)

data = f2.TrainerFixedC1(t0, i0, V0, socs0, C1)
plotx = []
ploty = []
for i in data:
    if i[4] == -2.5:
        plotx.append(i[5])
        ploty.append(i[1])
plt.scatter(plotx,ploty)

plt.scatter(plotx, ploty)

data = f2.TrainerFixedC1(t20, i20, V20, socs20, C1)
plotx = []
ploty = []
for i in data:
    if i[4] == -2.5:
        plotx.append(i[5])
        ploty.append(i[1])
plt.scatter(plotx,ploty)



plt.show()
**********************************************"""

"""
Part 2cc
"""
# """



#Plot the following conditions
SOCplot = 60
currentPlot = -2.5
R1_60SOC_n2_5 = []
T = np.linspace(0,40,3) + 273.15
# Get the data out of each array:
for j in parameters_0deg:
    if j[5] == SOCplot and j[4] == currentPlot:
        R1_60SOC_n2_5.append(j[1])

for j in parameters_20deg:
    if j[5] == SOCplot and j[4] == currentPlot:
        R1_60SOC_n2_5.append(j[1])

for j in parameters_40deg:
    if j[5] == SOCplot and j[4] == currentPlot:
        R1_60SOC_n2_5.append(j[1])

# plt.scatter(T, R1_60SOC_n2_5)
# plt.show()

# """


"""
Part 2cd
"""
# """
# Fitting the curve to the data:
E_R1_60 = f2.arrfitter(T, R1_60SOC_n2_5, 1)
Tfine = np.arange(270,320,0.05)
R1_pred_60 = f2.arrhenius(Tfine, R1_60SOC_n2_5[1], E_R1_60, 293.15)
# plt.plot(Tfine,R1_pred_60, label = 'SOC = 60%')
#
# plt.show()
# """
#
# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
# fig.set_facecolor('white')
#
# ax.plot(Tfine, R1_pred_60,linestyle = '--',c='cornflowerblue',zorder=10, clip_on=False, label = r'Fitted Arrhenius function')
# ax.scatter(T, R1_60SOC_n2_5, c='r',marker='o',zorder=10, clip_on=False,label = r"$R_1$ at 60\% $SOC$ at $-2.5\:A$")
#
#
# ax.set_xlabel(r"$T$ [$^{\circ} C$]",fontsize=14)
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
#
# plt.show()



"""
Part2ce
"""
# E_R1_60 = -30000


SOCplot = 30
currentPlot = -2.5
R1_30SOC_n2_5 = []
T = np.linspace(0,40,3) + 273.15
for j in parameters_0deg:
    if j[5] == SOCplot and j[4] == currentPlot:
        R1_30SOC_n2_5.append(j[1])

for j in parameters_20deg:
    if j[5] == SOCplot and j[4] == currentPlot:
        R1_30SOC_n2_5.append(j[1])

for j in parameters_40deg:
    if j[5] == SOCplot and j[4] == currentPlot:
        R1_30SOC_n2_5.append(j[1])


#
R1_pred_30 = f2.arrhenius(Tfine, R1_30SOC_n2_5[1], E_R1_60, 293.15)
# plt.plot(Tfine,R1_pred_30, label = 'Predicted at 30% SOC')
# plt.scatter(T, R1_30SOC_n2_5, marker = 'x', label = 'From Training Data')
# plt.legend()
# plt.show()



SOCplot = 90
currentPlot = -2.5
R1_90SOC_n2_5 = []
T = np.linspace(0,40,3) + 273.15
for j in parameters_0deg:
    if j[5] == SOCplot and j[4] == currentPlot:
        R1_90SOC_n2_5.append(j[1])

for j in parameters_20deg:
    if j[5] == SOCplot and j[4] == currentPlot:
        R1_90SOC_n2_5.append(j[1])

for j in parameters_40deg:
    if j[5] == SOCplot and j[4] == currentPlot:
        R1_90SOC_n2_5.append(j[1])


# Tfine = np.arange(270,320,0.05)
R1_pred_90 = f2.arrhenius(Tfine, R1_90SOC_n2_5[1], E_R1_60, 293.15)
# plt.plot(Tfine,R1_pred_90, label = 'Predicted at 90% SOC')
# plt.scatter(T, R1_90SOC_n2_5, marker = 'x', label = 'From Training Data')
# plt.legend()
# plt.show()


"""
Fitting the big equation (last part of c)
"""
# """
#Making an array of the temperatures, and combining all the parameters into one array
fullTemps = np.concatenate((np.full(np.shape(socs0), 0), np.full(np.shape(socs20), 20), np.full(np.shape(socs40), 40))) + 273.15
fullParameters = np.concatenate((parameters_0deg, parameters_20deg, parameters_40deg))



#Double up the low current resistance so better prediction there
for i in range(0,len(fullParameters)):
    if fullParameters[i,4] == -0.5:
        # fullParameters[i,4] = 0
        fullParameters = np.vstack((fullParameters,np.array([fullParameters[i,0], fullParameters[i,1], fullParameters[i,2], fullParameters[i,3], 0.5, fullParameters[i,5], fullParameters[i,6]])))
        fullTemps = np.hstack((fullTemps,fullTemps[i]))


X = np.array(([fullParameters[:,4],fullTemps]))
R1Params = curve_fit(f2.R1iT, X, fullParameters[:,1])[0]
# print(R1Params)
#
# predR1 = []
# for i in range(0,len(fullParameters)):
#     Xi = np.array([fullParameters[i,4],fullTemps[i]])
#     predR1.append(f2.R1iT(Xi,R1Params[0],R1Params[1],R1Params[2],R1Params[3],R1Params[4]))
#
# # plt.scatter(fullParameters[:,4],predR1)
# plt.scatter(fullParameters[:,4],fullParameters[:,1])
# plt.scatter(fullParameters[:,4],predR1)
# plt.show()

# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
# fig.set_facecolor('white')

#
# iFine = np.linspace(-20,5,200)
# TFine =  np.linspace(273,313,200)
# # TFine = np.full((200),293.15)
# igrid,Tgrid = np.meshgrid(iFine,TFine)
# Xfine = np.array([igrid,Tgrid])
# R1pred = f2.R1iT(Xfine,R1Params[0],R1Params[1],R1Params[2],R1Params[3],R1Params[4],R1Params[5])
# # print(np.shape(R1pred))
# plt.plot(Xfine[0,:],R1pred)
# plt.scatter(fullParameters[:,4],fullParameters[:,1])
# R1pred[R1pred>0.04]=0.04
# plt.pcolor(iFine,TFine,R1pred)
#
# plt.xlabel('Current [$A$]')
# plt.ylabel('Temperature [$^{\circ} C$]')
# cbar = plt.colorbar()
# cbar.set_label('R1 [$\Omega$]')
# #
# plt.show()

# for i in range(0,len(parameters_0deg))
# """

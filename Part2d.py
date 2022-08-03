from Part2c import R1Params
from Part2b import E_R0, R0_20deg, C1
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import Part2Functions as f2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ECVModel as f1

"""
This file is for doing the new prediction
"""



df = pd.read_csv(r"C:\Users\thoma\OneDrive\Desktop\FCCTPython\FCTT 18-19 Data\Battery_Testing_Data.csv")
t = np.array(df['Time (s)'])[1:]
i = np.array(df['Current (mA)'])[1:]/1000
V = np.array(df['Voltage (V)'])[1:]
Temp = np.array(df['Temperature'])[1:]

#Importing the SOC/OCV data
df2 = pd.read_csv(r"C:\Users\thoma\OneDrive\Desktop\FCCTPython\FCTT 18-19 Data\SOC_OCV_MFCTT_2019.csv", delimiter=('\t'))
SOCDataSheet = np.array(df2['SOC'])/100
SOCDataSheetV = np.array(df2['Ecell/V'])


batteryCapacity = 2.5
Vm = f2.model2(i, t, 293, R1Params,  R0_20deg, E_R0, C1, SOCDataSheet, SOCDataSheetV, batteryCapacity , 0.85)
error = Vm - V

# Simple plots here

# plt.plot(t,Vm[0],label = 'pred')
# plt.plot(t,V,label='data')
# plt.legend()

# plt.plot(t, (i/5)+4)
# """

"""
Plotting in the nice format for the report below
"""

# start = 18100
# end = 18800
#
# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
# fig.set_facecolor('white')
#
#
# ax.plot(t[start:end],Vm[0][start:end],c='blue',zorder=10, clip_on=False, label = r'Predicted by updated model')
# ax.plot(t[start:end],V[start:end],c='orange',zorder=10, clip_on=False, label = r'Measured test data')
#
#
#
# ax.set_xlabel(r"Time [$s$]",fontsize=14)
# ax.set_ylabel(r"Voltage [$V$]", color="k", fontsize=14)
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
# ax.set_xlim([18100,18800])
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
# """
# # plt.savefig('plot2a.pdf',dpi=300, bbox_inches = "tight")
# plt.show()


# """
# Next section of code for the big triple plot
# """
#
#
#
# plt.rcParams['text.usetex'] = True
#
# fig, ax = plt.subplots(3,figsize=(6*1.25,4*1.25), tight_layout=True)
# fig.set_facecolor('white')
#
# start = 18000
# end = 23000
#
# ########################
# '''''
# model voltage vs data
# '''''
#
# ax[0].plot(t[start:end], Vm[0][start:end],c='b',zorder=10,linewidth='1') #clip_on=False)
# ax[0].plot(t[start:end], V[start:end],c='tab:orange',zorder=10,linewidth='1') #clip_on=False)#
#
#
# # axis labels,title,legend
# ax[0].set_title(r'\bf{Predicted \& Actual Voltage as functions of $t$, using updated model}',fontsize=10)
# ax[0].set_ylabel(r"$v(t)$ [V]", color="k", fontsize=10)
# ax[0].legend((r"Model Predicted", r"Battery Testing Data"),shadow=False,fontsize=7,loc='upper right')
#
# # ticks and framing
# ax[0].minorticks_on()
# ax[0].grid(which='major', linestyle='-', linewidth='0.5',color='silver')
# ax[0].grid(which='minor', linestyle=':', linewidth='0.5',color='silver')
# ax[0].set_xlim([18000,23000])
# ax[0].set_ylim([3,4.5])
#
#
# '''''
# current
# '''''
# #dashed max line
#
# y00 = [-15,-15]
# x0 = [18000,23000]
# ax[1].plot(x0,y00,linestyle=':',c='gray')
# ax[1].plot(t[start:end],i[start:end],c='g',linewidth='1',zorder=10,clip_on=False)
# ax[1].text(19000,-13,r' $\vert$\textbf{Max}$\vert$ \textbf{= 15 A}',fontsize='8',c='dimgray')
#
# # axis labels and title
# ax[1].set_title(r'\bf{Current as a function of $t$}',fontsize=10)
# ax[1].set_ylabel(r"$i(t)$ [A]", color="k", fontsize=10)
#
# # ticks and framing
# ax[1].minorticks_on()
# ax[1].grid(which='major', linestyle='-', linewidth='0.5',color='silver')
# ax[1].grid(which='minor', linestyle=':', linewidth='0.5',color='silver')
# ax[1].set_xlim([18000,23000])
# ax[1].set_ylim([-16,14])
#
# '''''
# absolute error
# '''''
# abs_error = (Vm[0][start:end] - V[start:end])
# ax[2].plot(t[start:end], abs_error,c='darkviolet',linewidth='1',zorder=10) #clip_on=False)
#
# tn = t[start:end]
# errorn = np.abs(abs_error)
# print('error time ' + str(tn[np.argmax(errorn)]))
#
# # plotting dashed line
# x0 = [18000,23000]
# y0 = [max(abs_error),max(abs_error)]
# ax[2].plot(x0,y0,linestyle=':',c='gray')
#
# # axis labels and title
# ax[2].set_xlabel(r"$t$ [s]",fontsize=14)
# ax[2].set_ylabel(r"Absolute Error [V]", color="k", fontsize=10)
# ax[2].set_title(r'\bf{Absolute Error between Predicted \& Actual Voltage as a function of $t$}',fontsize=10)
# ax[2].text(18500,0.35,r'$\vert$\textbf{Max}$\vert$ \textbf{= 0.5454 V}',fontsize='8',c='dimgray')
# print(max(abs_error))
#
# # ticks and framing
# ax[2].minorticks_on()
# ax[2].grid(which='major', linestyle='-', linewidth='0.5',color='silver')
# ax[2].grid(which='minor', linestyle=':', linewidth='0.5',color='silver')
# ax[2].set_xlim([18000,23000])
# # ax[2].set_ylim([-0.2,0.23])
# plt.show()

# plt.savefig('errormultiplot_ABS.pdf',dpi=300, bbox_inches = "tight")

abs_error = (Vm[0] - V)
start = 18100
end = 23000

abs_error[abs_error>0.075] = 0.075
abs_error[abs_error<-0.075] = -0.075

plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
fig.set_facecolor('white')


ax.plot(t[start:end],abs_error[start:end],c='blue', clip_on=False)



ax.set_xlabel(r"Time [$s$]",fontsize=14)
ax.set_ylabel(r"Voltage error [$V$]", color="k", fontsize=14)


"""""
Gridlines & ticks
"""""
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='0.5',color='silver')
ax.grid(which='minor', linestyle=':', linewidth='0.5',color='silver')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

"""""
Framing
"""""
ax.set_ylim([-0.075,0.075])
ax.set_xlim([start,end])

"""""
Legend
"""""
# handles,labels = ax.get_legend_handles_labels()
#
# handles = [handles[1], handles[0]]
# labels = [labels[1], labels[0]]
# ax.legend(handles,labels,handlelength=1.5,shadow=True,fontsize=12,loc='upper right')

"""""
saving high res
"""
# plt.savefig('plot2a.pdf',dpi=300, bbox_inches = "tight")
plt.show()

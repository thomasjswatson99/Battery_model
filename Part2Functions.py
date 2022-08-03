import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import ECVModel as f1

def csvOpener(name):
    df = pd.read_csv(name)

    # print(df.columns)
    t = np.array(df['Time (s)'])
    i = np.array(df['Current (A)'])
    V = np.array(df['Voltage (V)'])

    return t, i, V


#This funciton is using the current data and correlating it with time and voltages
#to get info about the pulses out
def pulseLocator(iIn, tIn):
    startTimes = []
    startIndex = []
    endTimes = []
    endIndex = []
    iMags = []
    trigger = 0

    for j in range(0,len(tIn)):
        if iIn[j] != 0:
            if trigger == 0:
                startTimes.append(tIn[j])
                startIndex.append(j)
                iMags.append(iIn[j+1])
                trigger += 1
        else:
            if trigger > 0:
                endTimes.append(tIn[j-1])
                endIndex.append(j-1)
                trigger = 0


     #This function returns data about each pulse, where to get info about the
    #nth pulse, look for array[:,n]
    return np.array(([startTimes, endTimes, iMags, startIndex, endIndex]))

#This impedance calculator will return the values given the pulse data
#Will also calculate C1!
def impedances(tIn, VIn, iIn):
    #R0 from inital voltage drop
    R0i = (VIn[0] - VIn[1]) / iIn

    #This part of the code is to find the relaxation time
    #Expects relaxation after 3000 t steps
    j = 0
    while abs(VIn[j] - VIn[-1]) > 0.00001:
        j+=1

    dV = (VIn[0] - VIn[j])

    #Calculating R1 with R0
    R1 = (dV / iIn) - R0i

    #Calculating C1
    C1 = (tIn[j] - tIn[0]) / (4 * R1)

    return [R0i, R1, C1, dV, iIn]


#This function is similar to before but doesnt work out C1
def impedancesFixedC1(tIn, VIn, iIn, C1set):
    #R0 from inital voltage drop
    R0i = (VIn[0] - VIn[1]) / iIn
    #This part of the code is to find the relaxation time
    #Expects relaxation at the next pulse
    j = 0
    while abs(VIn[j] - VIn[-5]) > 0.00001:
        j+=1

    dV = (VIn[0] - VIn[j])
    # print(str(VIn[0])+ ' <> '+str(VIn[1]) + ' <> ' + str(VIn[j]))
    #Calculating R1 with R0
    R1 = (dV / iIn) - R0i

    # print(str(dV)+'<>'+str(R1))
    return [R0i, R1, C1set, dV, np.round(iIn,2)]


#This function will generate the arrays of battery data as required, using the
#pulse locator funciton as required
def batteryTrainerInitial(tIn, iIn, VIn, socsIn):

    #Using the funciton to find the data about pulses:
    pulseData = pulseLocator(iIn, tIn)

    #Will go through all of the pulses to do calculations on
    datapoints = []


    for j in range(0, len(pulseData[0,:])):
        #We will pass the data from the following index on to the function
        calcIndex = int(pulseData[4,j])

        #We need when the next pulse starts so we can get the steady state value
        if j != len(pulseData[0,:]) - 1:
            endIndex = int(pulseData[3,j+1]) - 1
        else:
            endIndex = len(tIn) - 1

        currentPulse = impedances(tIn[calcIndex:endIndex], VIn[calcIndex:endIndex], pulseData[2,j])
        currentPulse.append(socsIn[j])
        currentPulse.append(pulseData[2,j])
        datapoints.append(currentPulse)

    #This function will return the information in the following order:
    #R0, R1, C1, dV, i, state of charge, current
    #0   1   2   3   4      5               6
    return np.array(datapoints)

#This function accepts a set R0 and C1, as required
def revisedTrainer(iData, R0i, C1i):

    R60 = np.mean(iData[17:27,0])

    iData[:,4] = np.round(iData[:,4], 2)

    iData[:,0] = R0i

    iData[:,1] = (iData[:,3] / iData[:,4]) - R0i

    return  iData, R60


#For part 2, we are required to fix the value of C1, so including here a new
def TrainerFixedC1(tIn, iIn, VIn, socsIn, C1in):
    # plt.plot(tIn,VIn)
    # plt.show()
    #Using the funciton to find the data about pulses:
    pulseData = pulseLocator(iIn, tIn)
    #Will go through all of the pulses to do calculations on
    datapoints = []

    for j in range(0, len(pulseData[0,:])):
        #We will pass the data from the following index on to the function
        calcIndex = int(pulseData[4,j])

        #We need when the next pulse starts so we can get the steady state value
        if j != len(pulseData[0,:]) - 1:
            endIndex = int(pulseData[3,j+1]) - 1
        else:
            endIndex = len(tIn) - 1

        currentPulse = impedancesFixedC1(tIn[calcIndex:endIndex], VIn[calcIndex:endIndex], pulseData[2,j], C1in)
        currentPulse.append(socsIn[j])
        currentPulse.append(pulseData[2,j])
        datapoints.append(currentPulse)

    #This function will return the information in the following order:
    #R0, R1, C1, dV, i, state of charge, current
    return np.array(datapoints)

#This version of the modified gaussian function receives the coefficients as an
#array/tuple - used earlier but not useful for the curve_fit function
def modGaussa(x, R10A, coefs):
    return R10A * np.exp(-(x - coefs[0])**2 / coefs[1]) + coefs[2] * x + coefs[3]

def modGauss(x, R10A, b, c, d, e):
    return R10A * np.exp(-(x - b)**2 / c) + d * x + e

#Define a function which recieves the current and resistances
def gaussFitter(xin, yin):

    #Adding the smallest current resistnace again as the 0 value
    R10A = yin[np.argmin(np.abs(xin))]
    xin = np.concatenate((xin[:5], np.array([0]), xin[5:]))
    yin = np.concatenate((yin[:5], np.array([R10A]), yin[5:]))


    #Define our gaussian (ish) function
    #redefined in here so R10A can be set
    def gauss(xt, b, c, d, e):
        y = R10A * np.exp(- (xt - b)**2 / c) +d * xt + e
        return y

    #Using the fitting function from scipy
    parameters, covariance =  curve_fit(gauss, xin, yin)
    xfine = np.linspace(-20,5,200)

    return parameters


#Thhis function is for finding the fitting parameter of the arrhenius
def arrfitter(x, y, T0pos):
    T0 = x[T0pos]
    RT0 = y[T0pos]

    #Defining an arrhenius function in here again so RT0 can be set
    def arr(Tf, E):
        return RT0 * np.exp((-E / 8.3114) * ((1 / Tf) - (1 / T0)))
    parameters, covariance =  curve_fit(arr, x, y)

    return parameters[0]

#This function takes the fit parameters, and gives out the predicted R1
def arrhenius(x, RT0, E, T0):
    return RT0 * np.exp((-E / 8.3114) * ((1 / x) - (1 / T0)))

# This function can sort arrays before theyre fitted and average when there are
#two pulses of 4A
def pulseCleaner(Xarr, Yarr):
    sortIndex = np.argsort(Xarr)
    Xarr = Xarr[sortIndex]
    Yarr = Yarr[sortIndex]
    if len(Xarr) == 9:
        #Averaging the two pulses where its 4
        Xarr[-2] = np.mean(Xarr[-2:])
        Yarr[-2] = np.mean(Yarr[-2:])
        Xarr = Xarr[:-1]
        Yarr = Yarr[:-1]
    return Xarr, Yarr


def R1function(Tcoefs, E, I, T, a):
    return modGauss(I, 1, Tcoefs) * arrhenius(T, 1, E, 293.15) * a

#R1 as a function of temp and current. Xin as a 2D variable for the curve_fit algorithm
def R1iT(Xin, b, c, d, e, Ein, a):
    # Xin[0] = np.abs(Xin[0])
    return (np.exp(-(Xin[0] - b)**2 / c) + d * Xin[0] + e) * np.exp((Ein / 8.31) * ((1 / Xin[1]) - (1 / 293.15))) * 0.8 + a


#New function!
#Should recieve current time history, and a start SOC, then return a voltage!
def model2(i, t, temp, R1coefs, R0_20deg, E_R0, C1, SOCDataSheet, SOCDataSheetV, batCap, initialSOC):
    #This function will first get the simhistory for SOC (z)
    z = f1.SOCSimple(i, t, batCap, SOC = initialSOC)

    #will go iteratively through the time history data
    v = np.zeros(np.shape(t))
    v[0] = f1.SOCtoOCV(initialSOC, SOCDataSheet, SOCDataSheetV)
    iR1 = 0
    iR1l = [0]
    R1l=[0]

    iTX = np.array([0,temp])
    R1 = R1iT(iTX, R1coefs[0],R1coefs[1],R1coefs[2],R1coefs[3],R1coefs[4],R1coefs[5])

    for j in range(1,len(t)):
        #Get R1 and dt
        R0 = arrhenius(temp, R0_20deg, E_R0, 293.15)
        dt = t[j] - t[j-1]

        iR1 = np.exp(-(dt) / (R1 * C1)) * iR1 + (1 - np.exp(-(dt) / (R1 * C1))) * i[j-1]

        iTX = np.array([iR1,temp])
        R1 = R1iT(iTX, R1coefs[0],R1coefs[1],R1coefs[2],R1coefs[3],R1coefs[4],R1coefs[5])

        v[j] = f1.SOCtoOCV(z[j], SOCDataSheet, SOCDataSheetV) + R1 * iR1 + R0 * i[j-1]
        # R1l.append(R1)
        # iR1l.append(float(iR1))
    return v, z

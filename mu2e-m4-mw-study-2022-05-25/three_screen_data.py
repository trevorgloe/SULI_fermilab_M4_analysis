## script to perform the 3-screen method on beam data
# beam data taken from M4 beam line at fermilab's muon campus
# part of a SULI summer 2022 research project
# Author: Trevor Loe

import plot_fnc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

datanum = 3     # 1: screens 924,926,927    2: screens 926,927,930      3: screens 927,930,932

col = ['alpha','beta','emit']

curdr = os.getcwd()
csvpath = os.path.join(curdr,'three_screen_data_results.csv')
csv_exists = os.path.exists(csvpath)

if csv_exists:
    print('loading previous 3-screen data...')
    datacsv = pd.read_csv(csvpath,index_col=0)
else:
    print('No previous data file found. Create data file...')
    datacsv = pd.DataFrame(columns=col)

figpath = os.path.join(curdr,'three_screen_figures')

## data from file
if datanum==1:
    alldata = plot_fnc.read_scan_beamdata('data-mu2e-m4-mw924-mw926-mw927-2022-05-25.csv')
elif datanum==2:
    alldata = plot_fnc.read_scan_beamdata('data-mu2e-m4-mw926-mw927-mw930-2022-05-25.csv')
elif datanum==3:
    alldata = plot_fnc.read_scan_beamdata('data-mu2e-m4-mw927-mw930-mw932-2022-05-25.csv')

# starting message
if datanum==1:
    mesg = 'Data used is from screens MW924, MW926, and MW927'
elif datanum==2:
    mesg = 'Data used is from screens MW926, MW927, and MW930'
elif datanum==3:
    mesg = 'Data used is from screens MW927, MW930, and MW932'

# data names
if datanum==1:
    datanames = ['MW924','MW926','MW927']
elif datanum==2:
    datanames = ['MW926','MW927','MW930']
elif datanum==3:
    datanames = ['MW927','MW930','MW932']

# distances
if datanum==1:
    d = [1.52993,8.76993,14.86993]      # distance between Q924 and detectors MW924, MW926, and MW927
elif datanum==2:
    d = [0.483,6.583,30.453]            # distance between Q926 and detectors MW926, MW927 and MW930
elif datanum==3:
    d = [2.05755,25.92755,48.45755]       # distance between Q927 and detectors MW927, MW930, and MW932

# f.write(mesg+'\n')
print(mesg)


####### 3 screen method for x
print('in x:')
data1 = plot_fnc.extract_rowx(alldata,'sample_0_0')
data2 = plot_fnc.extract_rowx(alldata,'sample_1_0')
data3 = plot_fnc.extract_rowx(alldata,'sample_2_0')

# # check how the data is cutoff
# newdata,newx = plot_fnc.cutoff_noise(data927,np.linspace(-24,24,len(data927)))
# figtest = plt.figure()
# plt.plot(newx,newdata,'o')
# plt.title('cutoff data')

# calculate the sigmas for each of the datasets
# A,x0,sig1,err1 = plot_fnc.fit_guass(data1)
# A,x0,sig2,err2 = plot_fnc.fit_guass(data2)
# A,x0,sig3,err3 = plot_fnc.fit_guass(data3)

# print(sig1)
fig1 = plt.figure()
sig1,err1 = plot_fnc.fit_plot_guass(data1,datanames[0]+'x')
plt.savefig(os.path.join(figpath,datanames[0]+'x'))

fig2 = plt.figure()
sig2,err2 = plot_fnc.fit_plot_guass(data2,datanames[1]+'x')
plt.savefig(os.path.join(figpath,datanames[1]+'x'))

fig3 = plt.figure()
sig3,err3 = plot_fnc.fit_plot_guass(data3,datanames[2]+'x')
plt.savefig(os.path.join(figpath,datanames[2]+'x'))

print('Calculated sigmas from fits')
print(datanames[0]+': '+str(sig1)+' +/- '+str(err1[2]))
print(datanames[1]+': '+str(sig2)+' +/- '+str(err2[2]))
print(datanames[2]+': '+str(sig3)+' +/- '+str(err3[2]))

# plt.show()
sig3 = sig3 - err3[2]
sig2 = sig2 - err2[2]
sig1 = sig1 - err1[2]

sig1 = sig1/1000
sig2 = sig2/1000
sig3 = sig3/1000

# test to try to match other results
# sig1 = 0.00098
# sig2 = 0.0022
# sig3 = 0.0036

# distances
d1 = d[0]
d2 = d[1]
d3 = d[2]

print('using scipy')
# sigmas calculated are in millimeters, convert to meters before doing 3-screen method
alphax1,betax1,epsx1,ier = plot_fnc.thrsc(sig1,sig2,sig3,d1,d2,d3)

print('Data for file '+str(datanum)+' 3-screen method')
print('alpha = '+str(alphax1))
print('beta = '+str(betax1))
print('emit = '+str(epsx1))

# plt.show()
row = pd.DataFrame(np.array([[alphax1,betax1,epsx1]]),index=['file'+str(datanum)+' x '+'scipy'],columns=col)
datacsv = pd.concat([datacsv,row])

# using alternate implementation
print('using gekko')

# handle errors from not converging
try:
    alphax2,betax2,epsx2 = plot_fnc.thrsc_gek(sig1,sig2,sig3,d1,d2,d3)
except Exception:
    print('gekko did not converge')
    alphax2=[float('nan')]
    betax2=[float('nan')]
    epsx2=[float('nan')]

print('Data for file '+str(datanum)+' 3-screen method')
print('alpha = '+str(alphax2))
print('beta = '+str(betax2))
print('emit = '+str(epsx2))

# plt.show()
row = pd.DataFrame(np.array([[alphax2[0],betax2[0],epsx2[0]]]),index=['file'+str(datanum)+' x '+'gekko'],columns=col)
datacsv = pd.concat([datacsv,row])



###### 3-screen method in y
print('in y:')
data1 = plot_fnc.extract_rowy(alldata,'sample_0_0')
data2 = plot_fnc.extract_rowy(alldata,'sample_1_0')
data3 = plot_fnc.extract_rowy(alldata,'sample_2_0')

# # check how the data is cutoff
# newdata,newx = plot_fnc.cutoff_noise(data927,np.linspace(-24,24,len(data927)))
# figtest = plt.figure()
# plt.plot(newx,newdata,'o')
# plt.title('cutoff data')

# calculate the sigmas for each of the datasets
# A,x0,sig1,err1 = plot_fnc.fit_guass(data1)
# A,x0,sig2,err2 = plot_fnc.fit_guass(data2)
# A,x0,sig3,err3 = plot_fnc.fit_guass(data3)

# print(sig1)
fig1 = plt.figure()
sig1,err1 = plot_fnc.fit_plot_guass(data1,datanames[0]+'y')
plt.savefig(os.path.join(figpath,datanames[0]+'y'))

fig2 = plt.figure()
sig2,err2 = plot_fnc.fit_plot_guass(data2,datanames[1]+'y')
plt.savefig(os.path.join(figpath,datanames[1]+'y'))

fig3 = plt.figure()
sig3,err3 = plot_fnc.fit_plot_guass(data3,datanames[2]+'y')
plt.savefig(os.path.join(figpath,datanames[2]+'y'))

print('Calculated sigmas from fits')
print(datanames[0]+': '+str(sig1)+' +/- '+str(err1[2]))
print(datanames[1]+': '+str(sig2)+' +/- '+str(err2[2]))
print(datanames[2]+': '+str(sig3)+' +/- '+str(err3[2]))

# plt.show()

sig1 = sig1/1000
sig2 = sig2/1000
sig3 = sig3/1000

# test to try to match other results
# sig1 = 0.00098
# sig2 = 0.0022
# sig3 = 0.0036

# distances
d1 = d[0]
d2 = d[1]
d3 = d[2]

print('using scipy')
# sigmas calculated are in millimeters, convert to meters before doing 3-screen method
alphay1,betay1,epsy1,ier = plot_fnc.thrsc(sig1,sig2,sig3,d1,d2,d3)

print('Data for file '+str(datanum)+' 3-screen method')
print('alpha = '+str(alphay1))
print('beta = '+str(betay1))
print('emit = '+str(epsy1))

# plt.show()
row = pd.DataFrame(np.array([[alphay1,betay1,epsy1]]),index=['file'+str(datanum)+' y '+'scipy'],columns=col)
datacsv = pd.concat([datacsv,row])

# using alternate implementation
print('using gekko')

#handle errors from not converging
try:
    alphay2,betay2,epsy2 = plot_fnc.thrsc_gek(sig1,sig2,sig3,d1,d2,d3)
except Exception:
    print('gekko did not converge')
    alphay2=[float('nan')]
    betay2=[float('nan')]
    epsy2=[float('nan')]

print('Data for file '+str(datanum)+' 3-screen method')
print('alpha = '+str(alphay2))
print('beta = '+str(betay2))
print('emit = '+str(epsy2))

# plt.show()
row = pd.DataFrame(np.array([[alphay2[0],betay2[0],epsy2[0]]]),index=['file'+str(datanum)+' y '+'gekko'],columns=col)
datacsv = pd.concat([datacsv,row])

# save csv 
datacsv.to_csv(csvpath)

plt.show()


## Script to load in data outputted from G4beamline and calculate twist paramters
# uses the 3-screen method
# generalized from three_screen_s1/s2/s3
# Uses a 'control' file to calculate the twist parameters directly from the distribution and 
# then calculates the (hypothetically) same twist parameters via the 3 screen method on the 
# data from 3 screens following the control screen
# Made as a test for the implementation of the 3 screen method
# beamline should be completely clear in between test screens (just drift between them)
# made for use of data from simulations of the M4 beamline on fermilab's muon campus
#
# Made as part of a SULI Summer 2022 research project
# Author: Trevor Loe

import plot_fnc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


datanum = 2         # 1: screens 924,926,927    2: screens 926,927,930      3: screens 927,930,932
# make sure you have the right data files in the same folder as this script

### Open file for writting data
# f = open('Simu_3screen.txt')
# f.write('Data calculated from simulated data from G4beamline'+'\n')
# stores all data in a pandas dataframe
# appends to dataframe if it exists
col = ['alpha','beta','emit']

curdr = os.getcwd()
csvpath = os.path.join(curdr,'three_screen_results.csv')
csv_exists = os.path.exists(csvpath)

if csv_exists:
    print('loading previous 3-screen data...')
    datacsv = pd.read_csv(csvpath,index_col=0)
else:
    print('No previous data file found. Create data file...')
    datacsv = pd.DataFrame(columns=col)

# initialize path for saving figures
figpath = os.path.join(curdr,'three_screen_figures')

# starting message
if datanum==1:
    mesg = 'Data used is from screens MW924, MW926, and MW927'
elif datanum==2:
    mesg = 'Data used is from screens MW926, MW927, and MW930'
elif datanum==3:
    mesg = 'Data used is from screens MW927, MW930, and MW932'

# f.write(mesg+'\n')
print(mesg)

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


#### First use control file
control_data = plot_fnc.read_data('Control.txt')

print('Control distribution:')
print('in x:')
(alphax,betax,epsx) = plot_fnc.twistx(control_data)

print('alpha = '+str(alphax))
print('beta = '+str(betax))
print('eps = '+str(epsx))

# add data to csv
# f.write('x parameters: alpha= '+str(alphax)+'\t'+'beta= '+str(betax)+)
# print(datacsv)
row = pd.DataFrame(np.array([[alphax,betax,epsx]]),index=[str(datanum)+'Control dist x'],columns=col)
# print(row)
datacsv = pd.concat([datacsv,row])
# print(datacsv)
# plot_fnc.save_3hist(control_data,'control')
figcx = plt.figure()
plot_fnc.disp_save_g4blhist(control_data['#x'].to_numpy(),figpath,'Controlx')

print('in y:')
(alphay,betay,epsy) = plot_fnc.twisty(control_data)

print('alpha = '+str(alphay))
print('beta = '+str(betay))
print('eps = '+str(epsy))

# add data to csv
row = pd.DataFrame(np.array([[alphay,betay,epsy]]),index=[str(datanum)+'Control dist y'],columns=col)
datacsv = pd.concat([datacsv,row])

figcy = plt.figure()
plot_fnc.disp_save_g4blhist(control_data['y'].to_numpy(),figpath,'Controly')


#### Compute twist parameters via 3-screen method in x data
print('3-screen method in x')
data1 = plot_fnc.read_data(datanames[0]+'.txt')
data2 = plot_fnc.read_data(datanames[1]+'.txt')
data3 = plot_fnc.read_data(datanames[2]+'.txt')

# distances
d1 = d[0]
d2 = d[1]
d3 = d[2]

xdata1 = data1['#x'].to_numpy()
xdata2 = data2['#x'].to_numpy()
xdata3 = data3['#x'].to_numpy()

# convert to meters (originally in mm)
# xdata1 = xdata1/1000
# xdata2 = xdata2/1000
# xdata3 = xdata3/1000

# plot_fnc.disp_save_g4blhist(xdata1,figpath,datanames[0]+'x')

# fit a guassian to the data to check against calculated std
figx1 = plt.figure()
sig1c,sig1f = plot_fnc.hist_guassfit_g4bl(xdata1,datanames[0]+'x')
print(os.path.join(figpath,datanames[0]+'x'))
plt.savefig(os.path.join(figpath,datanames[0]+'x'))

# compare sigmas
print('For data '+datanames[0])
print('Calculated sigma = '+str(sig1c))
print('Fitted sigma = '+str(sig1f))

figx2 = plt.figure()
sig2c,sig2f = plot_fnc.hist_guassfit_g4bl(xdata2,datanames[1]+'x')
plt.savefig(os.path.join(figpath,datanames[1]+'x'))

# compare sigmas
print('For data '+datanames[1])
print('Calculated sigma = '+str(sig2c))
print('Fitted sigma = '+str(sig2f))

figx3 = plt.figure()
sig3c,sig3f = plot_fnc.hist_guassfit_g4bl(xdata3,datanames[2]+'x')
plt.savefig(os.path.join(figpath,datanames[2]+'x'))

# compare sigmas
print('For data '+datanames[2])
print('Calculated sigma = '+str(sig3c))
print('Fitted sigma = '+str(sig3f))

# convert to m before passing into three screen script
calphax,cbetax,cepsx = plot_fnc.thrsc(sig1c/1000,sig2c/1000,sig3c/1000,d1,d2,d3)

print('alpha = '+str(calphax))
print('beta = '+str(cbetax))
print('emit = '+str(cepsx))

row = pd.DataFrame(np.array([[calphax,cbetax,cepsx]]),index=[datanames[0]+' x data'],columns=col)
datacsv = pd.concat([datacsv,row])


#### Compute twist parameters via 3-screen method in y data
print('3-screen method in y:')
# distances
d1 = d[0]
d2 = d[1]
d3 = d[2]

ydata1 = data1['y'].to_numpy()
ydata2 = data2['y'].to_numpy()
ydata3 = data3['y'].to_numpy()

# convert to meters (originally in mm)
# xdata1 = xdata1/1000
# xdata2 = xdata2/1000
# xdata3 = xdata3/1000

# plot_fnc.disp_save_g4blhist(xdata1,figpath,datanames[0]+'x')

# fit a guassian to the data to check against calculated std
figy1 = plt.figure()
sig1c,sig1f = plot_fnc.hist_guassfit_g4bl(ydata1,datanames[0]+'y')
print(os.path.join(figpath,datanames[0]+'y'))
plt.savefig(os.path.join(figpath,datanames[0]+'y'))

# compare sigmas
print('For data '+datanames[0])
print('Calculated sigma = '+str(sig1c))
print('Fitted sigma = '+str(sig1f))

figy2 = plt.figure()
sig2c,sig2f = plot_fnc.hist_guassfit_g4bl(ydata2,datanames[1]+'y')
plt.savefig(os.path.join(figpath,datanames[1]+'y'))

# compare sigmas
print('For data '+datanames[1])
print('Calculated sigma = '+str(sig2c))
print('Fitted sigma = '+str(sig2f))

figy3 = plt.figure()
sig3c,sig3f = plot_fnc.hist_guassfit_g4bl(ydata3,datanames[2]+'y')
plt.savefig(os.path.join(figpath,datanames[2]+'y'))

# compare sigmas
print('For data '+datanames[2])
print('Calculated sigma = '+str(sig3c))
print('Fitted sigma = '+str(sig3f))

# convert to m before passing into three screen script
calphay,cbetay,cepsy = plot_fnc.thrsc(sig1c/1000,sig2c/1000,sig3c/1000,d1,d2,d3)

print('alpha = '+str(calphay))
print('beta = '+str(cbetay))
print('emit = '+str(cepsy))

row = pd.DataFrame(np.array([[calphay,cbetay,cepsy]]),index=[datanames[0]+' y data'],columns=col)
datacsv = pd.concat([datacsv,row])


# save csv 
datacsv.to_csv(csvpath)

plt.show()
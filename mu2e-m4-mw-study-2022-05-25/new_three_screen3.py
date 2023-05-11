## file to do the 3-screen method specifically for the MW927, MW930 and MW932 file
# has specific changes in the data for those specific guassian fits

import plot_fnc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

col = ['alpha','beta','emit']

curdr = os.getcwd()
csvpath = os.path.join(curdr,'three_screen_data_file3.csv')
csv_exists = os.path.exists(csvpath)

if csv_exists:
    print('loading previous 3-screen data...')
    datacsv = pd.read_csv(csvpath,index_col=0)
else:
    print('No previous data file found. Create data file...')
    datacsv = pd.DataFrame(columns=col)

figpath = os.path.join(curdr,'new_three_screen_figures_file3')

## data from file
alldata = plot_fnc.read_scan_beamdata('data-mu2e-m4-mw927-mw930-mw932-2022-05-25.csv')

# starting message
mesg = 'Data used is from screens MW927, MW930, and MW932'

# data names
datanames = ['MW927','MW930','MW932']

# distances
d = [2.05755,25.92755,48.45755]            # distance between Q927 and detectors MW927, MW930, and MW932

# f.write(mesg+'\n')
print(mesg)


# # check how the data is cutoff
# newdata,newx = plot_fnc.cutoff_noise(data927,np.linspace(-24,24,len(data927)))
# figtest = plt.figure()
# plt.plot(newx,newdata,'o')
# plt.title('cutoff data')

# calculate the sigmas for each of the datasets
# A,x0,sig1,err1 = plot_fnc.fit_guass(data1)
# A,x0,sig2,err2 = plot_fnc.fit_guass(data2)
# A,x0,sig3,err3 = plot_fnc.fit_guass(data3)


# print(basexdata)
# print(len(basexdata))


#############################################################################################################################################################
# data for x (horizontal)
#############################################################################################################################################################
print('in x:')
data1 = plot_fnc.extract_rowx(alldata,'sample_0_0')
data2 = plot_fnc.extract_rowx(alldata,'sample_1_0')
data3 = plot_fnc.extract_rowx(alldata,'sample_2_0')
basexdata = np.arange(-24,24,1)
## edit data to get rid of noise
# first data1
data1 = data1 - 0.045
# slice off first 16 points
newdata1 = data1[16:-1]
newxdata1 = basexdata[16:-1]
# slice off last 18
newdata1 = newdata1[0:-18]
newxdata1 = newxdata1[0:-18]

# print(data1)
# print(newdata1)
# print(basexdata)
# print(newxdata1)

fig1 = plt.figure()
sig1,err1 = plot_fnc.new_guass_fit(newxdata1,newdata1,basexdata,data1,datanames[0],True)
plt.savefig(os.path.join(figpath,datanames[0]+'x'))

# data2
data2 = data2 + 0.008
#slice off first 3 points
newdata2 = data2[3:-1]
newxdata2 = basexdata[3:-1]
#slice off last 22 pnts
newdata2 = newdata2[0:-22]
newxdata2 = newxdata2[0:-22]

fig2 = plt.figure()
sig2,err2 = plot_fnc.new_guass_fit(newxdata2,newdata2,basexdata,data2,datanames[1],True)
plt.savefig(os.path.join(figpath,datanames[1]+'x'))

# data3
#slice off first 6 points
newdata3 = data3[6:-1]
newxdata3 = basexdata[6:-1]
#slice off last 10 pnts
newdata3 = newdata3[0:-10]
newxdata3 = newxdata3[0:-10]

fig3 = plt.figure()
sig3,err3 = plot_fnc.new_guass_fit(newxdata3,newdata3,basexdata,data3,datanames[2],True)
plt.savefig(os.path.join(figpath,datanames[2]+'x'))

print('Calculated sigmas from fits')
print(datanames[0]+': '+str(sig1)+' +/- '+str(err1))
print(datanames[1]+': '+str(sig2)+' +/- '+str(err2))
print(datanames[2]+': '+str(sig3)+' +/- '+str(err3))

# do 3-screen method
sig1 = sig1/1000
sig2 = sig2/1000
sig3 = sig3/1000

err1 = err1/1000
err2 = err2/1000
err3 = err3/1000
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

print('Data for file '+'3'+' 3-screen method')
print('alpha = '+str(alphax1))
print('beta = '+str(betax1))
print('emit = '+str(epsx1))

# plt.show()
row = pd.DataFrame(np.array([[alphax1,betax1,epsx1]]),index=['file'+'3'+' x '+'scipy'],columns=col)
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

print('Data for file '+'3'+' 3-screen method')
print('alpha = '+str(alphax2))
print('beta = '+str(betax2))
print('emit = '+str(epsx2))

# plt.show()
row = pd.DataFrame(np.array([[alphax2[0],betax2[0],epsx2[0]]]),index=['file'+'3'+' x '+'gekko'],columns=col)
datacsv = pd.concat([datacsv,row])

# compute uncertainty
alphax_uncert,betax_uncert,epsx_uncert = plot_fnc.comp_3sc_uncert(sig1,sig2,sig3,err1/5,err2/5,err3/5,d1,d2,d3)
print('Uncertainties in twist:')
print('Uncertainty in alpha: '+str(alphax_uncert))
print('Uncertainty in beta: '+str(betax_uncert))
print('Uncertainty in eps: '+str(epsx_uncert))


#############################################################################################################################################################
# data for y (vertical)
#############################################################################################################################################################
print('in y:')
data1 = plot_fnc.extract_rowy(alldata,'sample_0_0')
data2 = plot_fnc.extract_rowy(alldata,'sample_1_0')
data3 = plot_fnc.extract_rowy(alldata,'sample_2_0')
baseydata = np.arange(-24,24,1)
## edit data to get rid of noise

# first data1
data1 = data1 - 0.009
# slice off first 14 points
newdata1 = data1[14:-1]
newydata1 = baseydata[14:-1]
# slice off last 12
newdata1 = newdata1[0:-12]
newydata1 = newydata1[0:-12]

# print(data1)
# print(newdata1)
# print(basexdata)
# print(newxdata1)

fig1 = plt.figure()
sig1,err1 = plot_fnc.new_guass_fit(newydata1,newdata1,baseydata,data1,datanames[0],False)
plt.savefig(os.path.join(figpath,datanames[0]+'y'))

# data2
data2 = data2 + 0.006
#slice off first 14 points
newdata2 = data2[14:-1]
newydata2 = baseydata[14:-1]
#slice off last 14 pnts
newdata2 = newdata2[0:-4]
newydata2 = newydata2[0:-4]

fig2 = plt.figure()
sig2,err2 = plot_fnc.new_guass_fit(newydata2,newdata2,baseydata,data2,datanames[1],False)
plt.savefig(os.path.join(figpath,datanames[1]+'y'))

# data3
data3 = data3 - 0.0025
#slice off first 1 points
newdata3 = data3[1:-1]
newydata3 = baseydata[1:-1]
#slice off last 14 pnts
newdata3 = newdata3[0:-10]
newydata3 = newydata3[0:-10]

fig3 = plt.figure()
sig3,err3 = plot_fnc.new_guass_fit(newydata3,newdata3,baseydata,data3,datanames[2],False)
plt.savefig(os.path.join(figpath,datanames[2]+'y'))

print('Calculated sigmas from fits')
print(datanames[0]+': '+str(sig1)+' +/- '+str(err1))
print(datanames[1]+': '+str(sig2)+' +/- '+str(err2))
print(datanames[2]+': '+str(sig3)+' +/- '+str(err3))

# do 3-screen method
sig1 = sig1/1000
sig2 = sig2/1000
sig3 = sig3/1000

err1 = err1/1000
err2 = err2/1000
err3 = err3/1000
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

print('Data for file '+'3'+' 3-screen method')
print('alpha = '+str(alphay1))
print('beta = '+str(betay1))
print('emit = '+str(epsy1))

# plt.show()
row = pd.DataFrame(np.array([[alphay1,betay1,epsy1]]),index=['file'+'3'+' y '+'scipy'],columns=col)
datacsv = pd.concat([datacsv,row])

# using alternate implementation
print('using gekko')

# handle errors from not converging
try:
    alphay2,betay2,epsy2 = plot_fnc.thrsc_gek(sig1,sig2,sig3,d1,d2,d3)
except Exception:
    print('gekko did not converge')
    alphay2=[float('nan')]
    betay2=[float('nan')]
    epsy2=[float('nan')]

print('Data for file '+'3'+' 3-screen method')
print('alpha = '+str(alphay2))
print('beta = '+str(betay2))
print('emit = '+str(epsy2))

# plt.show()
row = pd.DataFrame(np.array([[alphay2[0],betay2[0],epsy2[0]]]),index=['file'+'3'+' y '+'gekko'],columns=col)
datacsv = pd.concat([datacsv,row])

# compute uncertainty
alphay_uncert,betay_uncert,epsy_uncert = plot_fnc.comp_3sc_uncert(sig1,sig2,sig3,err1/5,err2/5,err3/5,d1,d2,d3)
print('Uncertainties in twist:')
print('Uncertainty in alpha: '+str(alphay_uncert))
print('Uncertainty in beta: '+str(betay_uncert))
print('Uncertainty in eps: '+str(epsy_uncert))

datacsv.to_csv(csvpath)

plt.show()

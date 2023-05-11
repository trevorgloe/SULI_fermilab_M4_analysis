### Script to run the 3-screen algorithm on beam data from fermilab's M4 beam line
# picking a random sigma for the fit near the average fitted value for the distributions loaded
# outputs all alpha,beta, and epsilons into a csv file 

import plot_fnc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

datanum = 3     # 1: screens 924,926,927    2: screens 926,927,930      3: screens 927,930,932

col = ['alpha','beta','emit','sig1','sig2','sig3','converge']

curdr = os.getcwd()
csvpath = os.path.join(curdr,'three_screen_rand_results.csv')

print('initializing csv...')
datacsv = pd.DataFrame(columns=col)

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


#### Use x data
print('x data:')
data1 = plot_fnc.extract_rowx(alldata,'sample_0_0')
data2 = plot_fnc.extract_rowx(alldata,'sample_1_0')
data3 = plot_fnc.extract_rowx(alldata,'sample_2_0')


fig1 = plt.figure()
sig1,err1 = plot_fnc.fit_plot_guass(data1,datanames[0]+'x')

fig2 = plt.figure()
sig2,err2 = plot_fnc.fit_plot_guass(data2,datanames[1]+'x')

fig3 = plt.figure()
sig3,err3 = plot_fnc.fit_plot_guass(data3,datanames[2]+'x')

print('Calculated sigmas from fits')
print(datanames[0]+': '+str(sig1)+' +/- '+str(err1[2]))
print(datanames[1]+': '+str(sig2)+' +/- '+str(err2[2]))
print(datanames[2]+': '+str(sig3)+' +/- '+str(err3[2]))

# distances
d1 = d[0]
d2 = d[1]
d3 = d[2]

# start loop to run algorithm with different sigmas
Nx = 200         # number of runs in x

# initialize data array
vals = np.zeros([2*Nx,len(col)])
idx = []

cnt_sci = 0
cnt_gek = 0

for i in range(Nx):
    alpha1,beta1,eps1,sci_conv,alpha2,beta2,eps2,gek_conv,newsig1,newsig2,newsig3=plot_fnc.random_sig_solve(sig1,sig2,sig3,err1[2],err2[2],err3[2],d1,d2,d3)

    # print(temp)
    # col = ['alpha','beta','emit','sig1','sig2','sig3','converge']
    vals[2*i,:] = np.array([alpha1,beta1,eps1,newsig1,newsig2,newsig3,sci_conv])
    vals[2*i+1,:] = np.array([alpha2,beta2,eps2,newsig1,newsig2,newsig3,gek_conv])
    idx.append('scipy x '+str(i))
    idx.append('gekko x '+str(i))

    if sci_conv==1:
        cnt_sci = cnt_sci + 1

    if gek_conv==1:
        cnt_gek = cnt_gek + 1

    # print(i,end='\r')
    print('run '+str(i))
    print('scipy? '+str(sci_conv))
    print('gekko? '+str(gek_conv))

print('scipy converged '+str(cnt_sci)+' times')
print('gekko converged '+str(cnt_gek)+' times')

xdata = pd.DataFrame(vals,index=idx,columns=col)

datacsv = pd.concat([datacsv,xdata])


#### Use y data
print('y data:')
data1 = plot_fnc.extract_rowy(alldata,'sample_0_0')
data2 = plot_fnc.extract_rowy(alldata,'sample_1_0')
data3 = plot_fnc.extract_rowy(alldata,'sample_2_0')


fig1 = plt.figure()
sig1,err1 = plot_fnc.fit_plot_guass(data1,datanames[0]+'y')

fig2 = plt.figure()
sig2,err2 = plot_fnc.fit_plot_guass(data2,datanames[1]+'y')

fig3 = plt.figure()
sig3,err3 = plot_fnc.fit_plot_guass(data3,datanames[2]+'y')

print('Calculated sigmas from fits')
print(datanames[0]+': '+str(sig1)+' +/- '+str(err1[2]))
print(datanames[1]+': '+str(sig2)+' +/- '+str(err2[2]))
print(datanames[2]+': '+str(sig3)+' +/- '+str(err3[2]))

# distances
d1 = d[0]
d2 = d[1]
d3 = d[2]

# start loop to run algorithm with different sigmas
Ny = 200         # number of runs in y

# initialize data array
vals = np.zeros([2*Ny,len(col)])
idx = []

cnt_sci = 0
cnt_gek = 0

for i in range(Ny):
    alpha1,beta1,eps1,sci_conv,alpha2,beta2,eps2,gek_conv,newsig1,newsig2,newsig3=plot_fnc.random_sig_solve(sig1,sig2,sig3,err1[2],err2[2],err3[2],d1,d2,d3)

    # col = ['alpha','beta','emit','sig1','sig2','sig3','converge']
    vals[2*i,:] = np.array([alpha1,beta1,eps1,newsig1,newsig2,newsig3,sci_conv])
    vals[2*i+1,:] = np.array([alpha2,beta2,eps2,newsig1,newsig2,newsig3,gek_conv])
    idx.append('scipy y '+str(i))
    idx.append('gekko y '+str(i))

    if sci_conv==1:
        cnt_sci = cnt_sci + 1

    if gek_conv==1:
        cnt_gek = cnt_gek + 1

    # print(i,end='\r')
    print('run '+str(i))
    print('scipy? '+str(sci_conv))
    print('gekko? '+str(gek_conv))

print('scipy converged '+str(cnt_sci)+' times')
print('gekko converged '+str(cnt_gek)+' times')

ydata = pd.DataFrame(vals,index=idx,columns=col)

datacsv = pd.concat([datacsv,ydata])

datacsv.to_csv(csvpath)

plt.show()
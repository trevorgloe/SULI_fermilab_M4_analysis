### Compute twist parameters using the quad_scan technique for simulate data form 
# G4beamline
# must have all folders containing VD data for each run with a different gradient value 
# must have control distribution for comparison
# part of a SULI Summer 2022 research project
# Author: Trevor Loe

import plot_fnc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
from scipy.optimize import curve_fit


# # first calculate some basic parameters 
# E = 1.2817e-9       # particle energy in J
# c = 299792458       # speed of light in m/s
# m = 105.6583755*1.79e-30        # mass of muon (converted to kg from MeV)
# q = 1.60217663e-19      # charge of muon (Coulumbs)
# # use relation (p*c)^2 = E^2 - (m*c^2)^2
# p = 1/c*math.sqrt(E**2 - (m*c**2)**2)       # momentum of particles in kg*m/s
# print('momentum = '+str(p))

# mult = q/p
# K = B*mult and f = 1/(KL)


# better method for calculating kappa
p = 8.89        # momentum in GeV/c
mult = 0.2998/p
# K = G*mult and f = 1/(K*L)

L = 457.20e-3       # effective length of SQA quad from G4beamline

# d = 3.91371         # distance from Q925 to MW930
d = 33.45651            # distance from Q925 to MW930


### 25 samples, run through each one and calculate the focual lenght then make a data point 
scandata = np.zeros([25,4])

# get magnetic fields of quads
quad_params = pd.read_csv('quad_scan_param_data.csv',index_col=0)

# create array of the indices
idx = []
for i in range(25):
    idx.append(quad_params.index[i])
# print(idx)
col = ['focusing x','sig_sqr x','focusing y','sig_sqr y']
data = pd.DataFrame(scandata,index=idx,columns=col)

# print(data)
# print(quad_params.loc['sample_1_0','Gradient'])


#### do quad scan for x data and y data

# get parabola data into dataframe
for index in idx:
    grad = (quad_params.loc[index,'Gradient'])
    # shortgrad = round(grad,6)
    tempstr = str(grad)
    shortgrad = tempstr[0:7]
    # print(shortgrad)
    for dir in os.walk(os.getcwd()):
        flag = False
        if dir[0].find(str(shortgrad))==-1:
            continue
        else:
            dirname = dir[0]
            flag = True
        # print(flag)

    # print(dirname)
    # get data for screen MW930
    screendata = plot_fnc.read_data(dirname+'/MW930.txt')

    xdata = screendata['#x'].to_numpy()
    sigx = xdata.std()
    # print(sigx)
    sigx = sigx/1000
    data.loc[index,'sig_sqr x'] = (sigx**2)

    ydata = screendata['y'].to_numpy()
    sigy = ydata.std()
    print(sigy)
    sigy = sigy/1000
    data.loc[index,'sig_sqr y'] = (sigy**2)

    # compute focusing parameter 1+d/f for x and 1-d/f for y
    # B = quad_params.loc[index,'Field']
    K = grad*mult
    f = 1/(K*L)

    focx = 1 + d/f
    data.loc[index,'focusing x']=focx
    focy = 1 - d/f
    data.loc[index,'focusing y']=focy

print(data)

# #display fitted data
# print('in x: ')
# figx = plt.figure()
# alpha,beta,gamma,eps,alpha_uncert,beta_uncert,gamma_uncert,eps_uncert = plot_fnc.fit_plot_parabola(data['focusing x'].to_numpy(),data['sig_sqr x'].to_numpy(),1e-5*np.ones(25),d,True)
# # plt.plot(data['focusing x'],data['sig_sqr x'],'o')
# # plt.title('Quad scan in x')
# # plt.xlabel('$1 - d/f$')
# # plt.ylabel('$\\langle x \\rangle^2$')

# # figy = plt.figure()
# # plt.plot(data['focusing y'],data['sig_sqr y'],'o')
# # plt.title('Quad scan in y')
# # plt.xlabel('$1 + d/f$')
# # plt.ylabel('$\\langle y \\rangle^2$')

# print('Extracted twist parameters:')
# print('alpha = '+str(alpha)+' +/- '+str(alpha_uncert))
# print('beta = '+str(beta)+' +/- '+str(beta_uncert))
# print('gamma = '+str(gamma)+' +/- '+str(gamma_uncert))
# print('eps = '+str(eps)+' +/- '+str(eps_uncert))


print('in y: ')
figx = plt.figure()
# alpha,beta,gamma,eps,alpha_uncert,beta_uncert,gamma_uncert,eps_uncert = plot_fnc.fit_plot_parabola(data['focusing y'].to_numpy(),data['sig_sqr y'].to_numpy(),1e-6*np.ones(25),d,False)
plot_fnc.new_quad_scan925(data['focusing y'].to_numpy(),data['sig_sqr y'].to_numpy(),1e-6*np.ones(25))

# print('Extracted twist parameters:')
# print('alpha = '+str(alpha)+' +/- '+str(alpha_uncert))
# print('beta = '+str(beta)+' +/- '+str(beta_uncert))
# print('gamma = '+str(gamma)+' +/- '+str(gamma_uncert))
# print('eps = '+str(eps)+' +/- '+str(eps_uncert))


### use initial distribution to get twist parameters
control_data = plot_fnc.read_data('Control.txt')

print('Control distribution:')
print('in x:')
(alphax,betax,epsx) = plot_fnc.twistx(control_data)

print('alpha = '+str(alphax))
print('beta = '+str(betax))
print('eps = '+str(epsx))


print('in y:')
(alphay,betay,epsy) = plot_fnc.twisty(control_data)

print('alpha = '+str(alphay))
print('beta = '+str(betay))
print('eps = '+str(epsy))


plt.show()






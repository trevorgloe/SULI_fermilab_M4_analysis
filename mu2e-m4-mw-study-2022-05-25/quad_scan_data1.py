### executes the quad scan method on beam data taken from fermilab's muon campus M4 beamline
# uses 2 data csv files that must be in the same directory as this file

import plot_fnc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

datanum = 1             # 1 - data for q925, 2 - data for q930

if datanum==1:
    alldata = plot_fnc.read_scan_beamdata('data-mu2e-m4-scan-dq925-2022-05-25.csv')
elif datanum==2:
    alldata = plot_fnc.read_scan_beamdata('data-mu2e-m4-scan-dq930-2022-05-25.csv')


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
# K = G*mult where G is the gradient in T/m

L = 457.20e-3       # effective length of SQA quad from G4beamline

d = 33.45651            # distance from Q925 to MW930

dir = os.getcwd()
figpath = os.path.join(dir,'quad_scan_figures1')




# get sample names
# quad_params = alldata['data_label']

# # create array of the indices (current set up to label the data points by their averages)
# data_labels = alldata.loc[1:,'data_label']
# data_labels = data_labels[1:-1]
# data_labels = data_labels[1:103:4]
# labels = data_labels.to_list()
# print(labels)
# idx = []
# for i in range(26):
#     idx.append(labels[i])
# # print(idx)
# col = ['focusing x','sig_sqr x','focusing y','sig_sqr y']
# data = pd.DataFrame(scandata,index=idx,columns=col)



# get magnetic fields of quads
if datanum==1:
    quad_params = pd.read_csv('quad_scan_param_data_1.csv',index_col=0)
elif datanum==2:
    quad_params = pd.read_csv('quad_scan_param_data_2.csv',index_col=0)

# print(len(quad_params['Current']))
### 26 samples, run through each one and calculate the focual lenght then make a data point 
scandata = np.zeros([len(quad_params['Current']),16])
# create array of the indices
idx = []
for i in range(len(quad_params['Current'])):
    idx.append(quad_params.index[i])
# print(idx)
col = ['focusing x','sigx','errx','sig_sqr x','sig_sqr errx','focusing y','sigy','erry','sig_sqr y','sig_sqr erry','lcutx','rcutx','offsetx','lcuty','rcuty','offsety']
data = pd.DataFrame(scandata,index=idx,columns=col)

## define the lcut, rcut and offset for each fit (6x26) for q925 scan file
fitadj = np.array([
    [4,12,0,20,6,-0.025],
    [5,10,0,11,16,-0.035],
    [3,11,0,11,14,-0.045],
    [3,10,0,9,14,-0.045],
    [3,10,0,11,16,-0.035],
    [3,10,0,9,14,-0.045],
    [3,12,0,11,14,-0.045],
    [6,10,0,11,18,-0.045],
    [5,8,0,11,16,-0.055],
    [7,10,0,11,16,-0.055],
    [7,10,0,14,19,-0.065],
    [7,10,0,13,16,-0.065],
    [5,10,0,11,16,-0.075],
    [3,10,0,11,13,-0.085],
    [11,10,0,11,14,-0.095],
    [7,10,0,11,14,-0.095],
    [7,10,0,14,15,-0.085],
    [7,10,0,10,11,-0.095],
    [7,12,0,11,10,-0.115],
    [10,10,0,9,12,-0.105],
    [11,7,0,10,13,-0.105],
    [8,5,0,11,12,0.000000],
    [8,10,0,11,14,0],
    [6,10,0,10,12,0],
    [0,1,0,0,1,0],
    [0,1,0,0,1,0]
])

data.loc[:,'lcutx':'offsety'] = fitadj
# print(data)
# loop through all samples
for name in idx:
    print(name)
    xdata = plot_fnc.extract_rowx_scan(alldata,name)
    ydata = plot_fnc.extract_rowy_scan(alldata,name)

    basedata = np.arange(-24,24,1)
    # fit gauassians to get the sigmas
    figx = plt.figure()
    # sigx,errx = plot_fnc.fit_plot_guass(xdata,name+'x')
    sigx,errx = plot_fnc.guass_fit_wcutoff(basedata,xdata,int(data.loc[name,'lcutx']),int(data.loc[name,'rcutx']),float(data.loc[name,'offsetx']),name,True)
    plt.savefig(os.path.join(figpath,name+'x'))

    figy = plt.figure()
    sigy,erry = plot_fnc.guass_fit_wcutoff(basedata,ydata,int(data.loc[name,'lcuty']),int(data.loc[name,'rcuty']),float(data.loc[name,'offsety']),name,False)
    plt.savefig(os.path.join(figpath,name+'y'))

    plt.close('all')

    # convert all values to meters
    data.loc[name,'sigx'] = sigx/1000
    data.loc[name,'errx'] = errx/1000

    data.loc[name,'sigy'] = sigy/1000
    data.loc[name,'erry'] = erry/1000

    data.loc[name,'sig_sqr x'] = (sigx/1000)**2
    data.loc[name,'sig_sqr y'] = (sigy/1000)**2

    data.loc[name,'sig_sqr errx'] = 2*(sigx/1000)*errx/1000
    data.loc[name,'sig_sqr erry'] = 2*(sigy/1000)*erry/1000

    # # compute focusing parameter 1+d/f for x and 1-d/f for y
    # B = quad_params.loc[name,'Field']
    # K = B*mult
    # f = 1/(K*L)
    G = quad_params.loc[name,'Gradient']
    K = G*mult
    f = 1/(K*L)

    focx = 1 + d/f
    data.loc[name,'focusing x']=focx
    focy = 1 - d/f
    data.loc[name,'focusing y']=focy

# get rid of some outlier data
data.loc['sample_24_1','sig_sqr x'] = float('nan')
data.loc['sample_25_1','sig_sqr x'] = float('nan')

data.loc['sample_24_1','sig_sqr y'] = float('nan')
data.loc['sample_25_1','sig_sqr y'] = float('nan')

data.loc['sample_23_1','sig_sqr y'] = float('nan')
data.loc['sample_22_1','sig_sqr y'] = float('nan')
data.loc['sample_21_1','sig_sqr y'] = float('nan')
data.loc['sample_20_1','sig_sqr y'] = float('nan')

print(data)
data.to_csv('quad_scan_calcs.csv')
#display fitted data
# print('in x: ')
# figx = plt.figure()
# alpha,beta,gamma,eps,alpha_uncert,beta_uncert,gamma_uncert,eps_uncert = plot_fnc.fit_plot_parabola(data['focusing x'].to_numpy(),data['sig_sqr x'].to_numpy(),data['sig_sqr errx'].to_numpy(),d,True)
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
alpha,beta,gamma,eps,alpha_uncert,beta_uncert,gamma_uncert,eps_uncert = plot_fnc.new_quad_scan925(data['focusing y'].to_numpy(),data['sig_sqr y'].to_numpy(),data['sig_sqr erry'].to_numpy())
# plt.plot(data['focusing x'],data['sig_sqr x'],'o')
# plt.title('Quad scan in x')
# plt.xlabel('$1 - d/f$')
# plt.ylabel('$\\langle x \\rangle^2$')

# figy = plt.figure()
# plt.plot(data['focusing y'],data['sig_sqr y'],'o')
# plt.title('Quad scan in y')
# plt.xlabel('$1 + d/f$')
# plt.ylabel('$\\langle y \\rangle^2$')

print('Extracted twist parameters:')
print('alpha = '+str(alpha)+' +/- '+str(alpha_uncert))
print('beta = '+str(beta)+' +/- '+str(beta_uncert))
print('gamma = '+str(gamma)+' +/- '+str(gamma_uncert))
print('eps = '+str(eps)+' +/- '+str(eps_uncert))


plt.show()
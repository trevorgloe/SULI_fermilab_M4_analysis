## script to calculate magnet gradients for M4 beamline at fermilab Muon Campus
# data used if from data-mu2e-m4-base-beam-2022-05-25.csv and Quad Excitation Summary.xls
# Part of a summer 2022 SULI research project
# Author: Trevor Loe

import plot_fnc
import pandas as pd
import numpy as np

# store all current values in pandas dataframe
base_data = plot_fnc.read_scan_beamdata('data-mu2e-m4-base-beam-2022-05-25.csv')

currents = base_data.loc[0:1,'ps_setting_0':'ps_setting_24']
current_vals = currents.loc[1,:]
current_vals = current_vals.astype(float)
current_vals = current_vals.to_numpy()
current_labels = currents.loc[0,:]
# index = data[data['data_label_0']==samplename].index.values[0]
Idata = pd.DataFrame(current_vals,current_labels)

magnets_wdata = ['D:Q903','D:Q906','D:Q908','D:Q909','D:Q914','D:Q919','D:Q922','D:Q924','D:Q926','D:Q927','D:Q930','D:Q932']

# append magnet types to the dataframe
magnet_types = ['SQA','SQD','SQD','SQD','SQA','SQC','SQD','SQA','SQA','SQC','SQC','SQA','SQD','SQD','SQA','SQA','SQA','SQA','SQA','SQB','SQA','SQA','SQA','SQA','SQB']

Idata.columns = ['Current']
Idata['Mag type'] = magnet_types
print(Idata)

# create dataframe of coefficients for calcuating magnetic field
# each row is a set of coefficients for the same magnet type
# order of the rows goes SQA, SQB, SQC, SQD
coef_vals = np.array([[2.868060E-02,1.877096E-02,-2.597822E-06,2.748336E-08,-6.634871E-11],[4.526777E-02,2.588056E-02,-4.371694E-07,2.694751E-08,-7.492979E-11],[5.236741E-02,2.825189E-02,1.291008E-06,1.879701E-08,-6.520112E-11],[6.023389E-02,3.343848E-02,-5.212306E-07,3.319100E-08,-9.208405E-11]])

coef = pd.DataFrame(coef_vals,index=['SQA','SQB','SQC','SQD'],columns=['a0','a1','a2','a3','a4'])

print(coef)

# iterate through the magnets and calculate magnetic field using the coefficients
magfields = []
for index, row in Idata.iterrows():
    # print(row['Current'])
    curr = row['Current']
    if row['Mag type']=='SQA':
        avals = coef.loc['SQA',:].values
        
    elif row['Mag type']=='SQB':
        avals = coef.loc['SQB',:].values

    elif row['Mag type']=='SQC':
        avals = coef.loc['SQC',:].values

    elif row['Mag type']=='SQD':
        avals = coef.loc['SQD',:].values

    # calculate magnetic field from a coefficients
    mfield = avals[0] + avals[1]*curr + avals[2]*curr**2 + avals[3]*curr**3 + avals[4]*curr**4
    print(mfield)
    magfields.append(mfield)

# print(magfields)

# add calcualted magnetic field to dataframe
Idata['Field'] = magfields

print(Idata)

# add effective length and divide
lengths = []
grad = []
for index, row in Idata.iterrows():
    if row['Mag type']=='SQA':
        leng = 457.20e-3
        
    elif row['Mag type']=='SQB':
        leng = 640.08e-3

    elif row['Mag type']=='SQC':
        leng = 701.04e-3

    elif row['Mag type']=='SQD':
        leng = 828.04e-3

    lengths.append(leng)
    gradient = row['Field']/leng

    grad.append(gradient)

Idata['Eff Length'] = lengths
Idata['Gradient'] = grad

print(Idata)

f = open('gradientvalues.txt','a')
for index,row in Idata.iterrows():
    f.write(index + '\t' + str(row['Gradient'])+'\n')

f.close()
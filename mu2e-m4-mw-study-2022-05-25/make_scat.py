## file to just save a bunch of scatter plots 

import plot_fnc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

dir = os.getcwd()
figpath = os.path.join(dir,'scan_prof_figs')

alldata = plot_fnc.read_scan_beamdata('data-mu2e-m4-scan-dq925-2022-05-25.csv')

labels = alldata.loc[0:105,'data_label']
labels = labels[0:-1]
labels = labels[1:105:4]
labels = labels.values.tolist()
# print(labels)
alldata = alldata.set_index('data_label')
print(alldata)

for name in labels:
    # index = alldata[alldata['data_label']==name].index.values[0]
    # print(index)
    # temp = alldata.loc[[index]]
    # temp = alldata.loc[[index],'mw_reading_0_0':'mw_reading_0_95'].to_numpy()
    # temp = alldata.loc[name,'mw_reading_0_0':'mw_reading_0_95'].to_numpy()
    temp = alldata.loc[name,'mw_reading_0_0':'mw_reading_0_95'].to_numpy()
    # print(temp)
    vals = temp.astype(float)
    # x = np.linspace(0,96,96)
    x = np.linspace(0,len(vals),len(vals))
    # vals = vals[0]
    fig = plt.figure()
    # print(vals)
    plt.plot(x,vals,'o')
    plt.title(name)
    plt.xlabel('wire')
    plt.ylabel('intensity')

    plt.savefig(os.path.join(figpath,name))



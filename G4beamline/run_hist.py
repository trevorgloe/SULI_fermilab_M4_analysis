## Script to create histograms from ASCII file outputs of G4beamline
# for simulation of M4 beamline on fermilab muon campus

import plot_fnc
import pandas as pd
import matplotlib.pyplot as plt


# data1 = plot_fnc.read_data('VD_Diagnostic_001.txt')

# fig1 = plt.figure()
# plt.hist(data1['#x'],bins=30)

# plt.savefig('VD_Diagnostic_001_x')
# plt.show()
# data = plot_fnc.read_data('MW903.txt')

#open file to write standard deviations
f = open('allmeanstdg4bl.txt','a')
f.write('Standard deviations for all histograms generated \n')
f.write('Detector'+'\t'+'xstd'+'\t'+'ystd'+'\n')

names = ['MW903','MW906','MW908','MW910','MW910','MW914','MW919','MW922','MW924','MW926','MW927','MW930','MW932']

print(len(names))

for name in names:
    data = plot_fnc.read_data(name+'.txt')
    print(data.shape)

    (xstd, ystd) = plot_fnc.save_3hist(data,name)
    f.write(name+'\t'+str(xstd)+'\t'+str(ystd)+'\n')

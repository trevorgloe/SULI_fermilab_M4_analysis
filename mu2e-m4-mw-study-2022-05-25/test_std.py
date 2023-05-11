## script to test the functionality of computing the standard deviation of beam data
# Author: Trevor Loe

import plot_fnc
import matplotlib.pyplot as plt

name = 'data-mu2e-m4-mw924-mw926-mw927-2022-05-25.csv'
data = plot_fnc.read_scan_beamdata(name)

row = plot_fnc.extract_row(data,'sample_0_0')
# print(row)

fig1 = plt.figure()

plot_fnc.fit_plot_guass(row)

plt.show()
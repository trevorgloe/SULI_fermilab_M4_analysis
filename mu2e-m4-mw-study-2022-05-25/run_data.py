## Analyze data taken from the M4 beamline at fermilab's muon campus
# uses file plot_fnc.py
# part of a SULI 2022 summer project
# Author: Trevor Loe

from cgi import test
import plot_fnc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# name = 'data-mu2e-m4-scan-dq930-2022-05-25.csv'
# # name = 'data-mu2e-m4-mw924-mw926-mw927-2022-05-25.csv'

# data = plot_fnc.read_scan_beamdata(name)
# # print(data)
# # fig = plt.figure()
# # plot_fnc.make_beam_hist(data,'average_1')
# # plot_fnc.make_mw_beam_scatterplot(data,'average_1')
# # fig2 = plt.figure()
# # plot_fnc.make_beam_hist(data,'average_2')
# # plot_fnc.make_mw_beam_scatterplot(data,'average_2')
# # fig3 = plt.figure()
# # plot_fnc.make_beam_hist(data,'average_0')
# # plot_fnc.make_mw_beam_scatterplot(data,'average_0')
# # fig4 = plt.figure()
# # plot_fnc.make_scan_beam_hist(data,'average_15')
# # fig5 = plt.figure()
# # plot_fnc.make_scan_beam_hist(data,'average_29')
# # data.to_csv('test.csv')


# # names = ['data-mu2e-m4-base-background-2022-05-25.csv','data-mu2e-m4-base-beam-2022-05-25.csv','data-mu2e-m4-mw924-mw926-mw927-2022-05-25.csv','data-mu2e-m4-mw926-mw927-mw930-2022-05-25.csv','data-mu2e-m4-mw927-mw930-mw932-2022-05-25.csv','data-mu2e-m4-scan-dq925-2022-05-25.csv','data-mu2e-m4-scan-dq930-2022-05-25.csv','data-mu2e-p1p2m1m3dr-quads-2022-05-25.csv']

# # for name in names:
# #     data = plot_fnc.read_scan_beamdata(name)

# #     data.to_csv('reformatted_data/'+name)
# #     print('saved!')
# fig1 = plt.figure()
# plot_fnc.make_scan_beam_bar(data,'average_5')
# plt.show()

# test_row = data.loc[[121],'mw_reading_0_0':'mw_reading_0_95']
# print(data[data['data_label']=='sample_30_0'].index.values[0])
# numpy_data = test_row.to_numpy()

# test_row = test_row.astype(float)
# print(test_row)
# test_row = test_row.transpose()
# # numpy_data = test_row.to_numpy()
# # numpy_data = numpy_data[0]
# # print(numpy_data)
# print(test_row)
# fig = plt.figure()
# plt.hist(test_row,bins=15)
# # plt.scatter(np.linspace(0,1,len(numpy_data)),numpy_data)
# # test_row.plot()
# plt.show()

# create file for writing the means and standard deviations
f = open('allmeanstd.txt','a')
f.write('Mean and standard deviations for all histograms generated \n')
f.write('Detector'+'\t'+'xmean'+'\t'+'xstd'+'\t'+'ymean'+'\t'+'ystd'+'\n')

# create all histograms from data
# make sure all folders in data_histograms directory are deleted or moved somewhere else before running
# not using data fro the 'scan' files
names = ['data-mu2e-m4-mw924-mw926-mw927-2022-05-25.csv','data-mu2e-m4-mw926-mw927-mw930-2022-05-25.csv','data-mu2e-m4-mw927-mw930-mw932-2022-05-25.csv']

# create the plots for first file
name = names[0]
data1 = plot_fnc.read_scan_beamdata(name)

# first detector in file
detname = 'MW924'
# create folder for saving the figures
plot_fnc.make_hist_dir('file1'+detname)

# create plot for first sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_0_0','file1_'+detname+'_sample0','file1'+detname)
f.write('file1_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for second sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_0_1','file1_'+detname+'_sample1','file1'+detname)
f.write('file1_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for average of two sameples
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'average_0','file1_'+detname+'_avg','file1'+detname)
f.write('file1_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# second detector in file
detname = 'MW926'
# create folder for saving the figures
plot_fnc.make_hist_dir('file1'+detname)

# create plot for first sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_1_0','file1_'+detname+'_sample0','file1'+detname)
f.write('file1_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for second sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_1_1','file1_'+detname+'_sample1','file1'+detname)
f.write('file1_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for average of two sameples
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'average_1','file1_'+detname+'_avg','file1'+detname)
f.write('file1_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# third detector in file
detname = 'MW927'
# create folder for saving the figures
plot_fnc.make_hist_dir('file1'+detname)

# create plot for first sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_2_0','file1_'+detname+'_sample0','file1'+detname)
f.write('file1_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for second sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_2_1','file1_'+detname+'_sample1','file1'+detname)
f.write('file1_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for average of two sameples
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'average_2','file1_'+detname+'_avg','file1'+detname)
f.write('file1_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')


# create the plots for second file
name = names[1]
data2 = plot_fnc.read_scan_beamdata(name)

# first detector in file
detname = 'MW926'
# create folder for saving the figures
plot_fnc.make_hist_dir('file2'+detname)

# create plot for first sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_0_0','file2_'+detname+'_sample0','file2'+detname)
f.write('file2_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for second sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_0_1','file2_'+detname+'_sample1','file2'+detname)
f.write('file2_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for average of two sameples
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'average_0','file2_'+detname+'_avg','file2'+detname)
f.write('file2_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# second detector in file
detname = 'MW927'
# create folder for saving the figures
plot_fnc.make_hist_dir('file2'+detname)

# create plot for first sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_1_0','file2_'+detname+'_sample0','file2'+detname)
f.write('file2_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for second sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_1_1','file2_'+detname+'_sample1','file2'+detname)
f.write('file2_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for average of two sameples
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'average_1','file2_'+detname+'_avg','file2'+detname)
f.write('file2_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# third detector in file
detname = 'MW930'
# create folder for saving the figures
plot_fnc.make_hist_dir('file2'+detname)

# create plot for first sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_2_0','file2_'+detname+'_sample0','file2'+detname)
f.write('file2_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for second sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_2_1','file2_'+detname+'_sample1','file2'+detname)
f.write('file2_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for average of two sameples
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'average_2','file2_'+detname+'_avg','file2'+detname)
f.write('file2_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')


# create the plots for third file
name = names[2]
data3 = plot_fnc.read_scan_beamdata(name)

# first detector in file
detname = 'MW927'
# create folder for saving the figures
plot_fnc.make_hist_dir('file3'+detname)

# create plot for first sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_0_0','file3_'+detname+'_sample0','file3'+detname)
f.write('file3_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for second sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_0_1','file3_'+detname+'_sample1','file3'+detname)
f.write('file3_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for average of two sameples
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'average_0','file3_'+detname+'_avg','file3'+detname)
f.write('file3_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# second detector in file
detname = 'MW930'
# create folder for saving the figures
plot_fnc.make_hist_dir('file3'+detname)

# create plot for first sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_1_0','file3_'+detname+'_sample0','file3'+detname)
f.write('file3_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for second sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_1_1','file3_'+detname+'_sample1','file3'+detname)
f.write('file3_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for average of two sameples
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'average_1','file3_'+detname+'_avg','file3'+detname)
f.write('file3_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# third detector in file
detname = 'MW932'
# create folder for saving the figures
plot_fnc.make_hist_dir('file3'+detname)

# create plot for first sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_2_0','file3_'+detname+'_sample0','file3'+detname)
f.write('file3_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for second sample of data
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'sample_2_1','file3_'+detname+'_sample1','file3'+detname)
f.write('file3_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')

# create plot for average of two sameples
(xmean,xstd,ymean,ystd) = plot_fnc.save_2bar(data1,'average_2','file3_'+detname+'_avg','file3'+detname)
f.write('file3_'+detname + '\t' + str(xmean)+'\t'+str(xstd)+'\t'+str(ymean)+'\t'+str(ystd)+'\n')


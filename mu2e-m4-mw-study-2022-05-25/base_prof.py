## script to create figures for all the beam profiles from the base-beam data file
# saves all figures into a directory in teh same place as this file
# part of a SULI summer 2022 research project
# Author: Trevor Loe


import plot_fnc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib

alldata = plot_fnc.read_scan_beamdata('data-mu2e-m4-base-beam-2022-05-25.csv')

screennames = ['MW903','MW906','MW908','MW909','MW914','MW919','MW922','MW924','MW926','MW927','MW930','MW932']

names = ['MW903','MW906','MW908','MW910','MW910','MW914','MW919','MW922','MW924','MW926','MW927','MW930','MW932']

# set font sizes
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

dir = os.getcwd()
figpath = os.path.join(dir,'basebeam_figures')

f = open('basebeam_sigmas.txt','w')
f.write('Sigmas found for all beam profiles via guassian fits')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

for i in range(12):
    name = screennames[i]

    fsampname = 'sample_'+str(i)+'_0'
    ssampname = 'sample_'+str(i)+'_1'
    avgname = 'average_'+str(i)

    xvals = np.linspace(-24,24,48)

    # get data
    xdata1 = plot_fnc.extract_rowx(alldata,fsampname)
    ydata1 = plot_fnc.extract_rowy(alldata,fsampname)

    # get sigmas and save fit pictures
    fitx1 = plt.figure()
    sigx,x0,errx = plot_fnc.fit_plot_guass(xdata1,name+'sample0x')
    plt.savefig(os.path.join(figpath,name+'sample0x_fit'))
    fity1 = plt.figure()
    sigy,y0,erry = plot_fnc.fit_plot_guass(ydata1,name+'sample0y')
    plt.savefig(os.path.join(figpath,name+'sample0y_fit'))

    # put x and y on the same figure
    xdata1 = -(xdata1-np.max(xdata1))
    ydata1 = -(ydata1-np.max(ydata1))
    fig1 = plt.figure()
    plt.scatter(xvals,xdata1,color='blue',label='horizontal')
    plt.scatter(xvals,ydata1,color='green',label='vertical')
    plt.xlabel('x or y [mm]')
    plt.ylabel('Intensity')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,name+'sample0'))

    # write data to file
    f.write(name+'sample_0 x: \t'+str(sigx)+' +/- '+str(errx[2])+'\n')
    f.write(name+'sample_0 y: \t'+str(sigy)+' +/- '+str(erry[2])+'\n')


    xdata2 = plot_fnc.extract_rowx(alldata,ssampname)
    ydata2 = plot_fnc.extract_rowy(alldata,ssampname)

    # get sigmas and save fit pictures
    fitx2 = plt.figure()
    sigx,x0,errx = plot_fnc.fit_plot_guass(xdata2,name+'sample1x')
    plt.savefig(os.path.join(figpath,name+'sample1x_fit'))
    fity2 = plt.figure()
    sigy,y0,erry = plot_fnc.fit_plot_guass(ydata2,name+'sample1y')
    plt.savefig(os.path.join(figpath,name+'sample1y_fit'))

    # put x and y on the same figure
    xdata2 = -(xdata2-np.max(xdata2))
    ydata2 = -(ydata2-np.max(ydata2))
    fig2 = plt.figure()
    plt.scatter(xvals,xdata2,color='blue',label='horizontal')
    plt.scatter(xvals,ydata2,color='green',label='vertical')
    plt.xlabel('x or y [mm]')
    plt.ylabel('Intensity')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,name+'sample1'))

    # write data to file
    f.write(name+'sample_1 x: \t'+str(sigx)+' +/- '+str(errx[2])+'\n')
    f.write(name+'sample_1 y: \t'+str(sigy)+' +/- '+str(erry[2])+'\n')

    xdataavg = plot_fnc.extract_rowx(alldata,avgname)
    ydataavg = plot_fnc.extract_rowy(alldata,avgname)

    # get sigmas and save fit pictures
    fitx3 = plt.figure()
    sigx,x0,errx = plot_fnc.fit_plot_guass(xdataavg,name+'averagex')
    plt.savefig(os.path.join(figpath,name+'averagex_fit'))
    fity3 = plt.figure()
    sigy,y0,erry = plot_fnc.fit_plot_guass(ydataavg,name+'averagey')
    plt.savefig(os.path.join(figpath,name+'averagey_fit'))

    xshift = round(x0)
    yshift = round(y0)
    # put x and y on different figures
    xdataavg = -(xdataavg-np.max(xdataavg))
    ydataavg = -(ydataavg-np.max(ydataavg))
    # normalize data in y
    xdataavg = xdataavg/np.max(xdataavg)
    ydataavg = ydataavg/np.max(ydataavg)
    # shift data so the mean is in the center
#     print(xdataavg)
    xdataavg = plot_fnc.shift_data(xdataavg,-xshift)
    ydataavg = plot_fnc.shift_data(ydataavg,-yshift)
#     print(xdataavg)

    fig3x = plt.figure()
    plt.scatter(xvals,xdataavg,color='blue',label='horizontal')
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity')
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,name+'average_x'))

    fig3y = plt.figure()
    plt.scatter(xvals,ydataavg,color='green',label='vertical')
    plt.xlabel('y [mm]')
    plt.ylabel('Intensity')
#     plt.legend(loc='upper right')
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,name+'average_y'))

    # write data to file
    f.write(name+'average x: \t'+str(sigx)+' +/- '+str(errx[2])+'\n')
    f.write(name+'average y: \t'+str(sigy)+' +/- '+str(erry[2])+'\n')


    #### plot both G4beamline data along with real data for comparison
    ## first check if the name actually matches one of the G4beamline data points
    if name in names:
        # plot G4beamline data
        data = plot_fnc.read_data('G4beamline_base_data/'+name+'.txt')
        binsno = 48
        bins = np.linspace(-24,24,binsno)
        newbins = np.linspace(-24,24,binsno-1)
        xlimits = (-24,24)

        # x data histogram
        figtempx = plt.figure()
        xn,xbins,patches = plt.hist(data['#x'], bins=bins,rwidth=0.85)
        # figure labels and stuff
        # plt.xlabel('x [mm]')
        # plt.ylabel('Events')
        # plt.xlim(xlimits)
        # plt.title(name)

        # plt.savefig('histograms/x'+name)

        print('x std: ')
        print(data['#x'].std())

        # y data
        figtempy = plt.figure()
        yn,ybins,patches = plt.hist(data['y'], bins=bins,rwidth=0.85)
        # figure labels and stuff
        # plt.xlabel('y [mm]')
        # plt.ylabel('Events')
        # plt.xlim(xlimits)
        # plt.title(name)
        # print(xn)
        # print(bins)
        # create sactter plots with histogram data

        # normalize data
        xn = xn/np.max(xn)
        yn = yn/np.max(yn)

        xshift = -(np.argmax(xn) - 24)
        yshift = -(np.argmax(yn) - 24)

        xn = plot_fnc.shift_data(xn,xshift)
        yn = plot_fnc.shift_data(yn,yshift)

        figbothx = plt.figure()

        #G4beamline data
        plt.plot(newbins,xn,'o',color='green',marker='^',label='model')

        #Real data
        plt.scatter(xvals,xdataavg,color='blue',label='data')
        plt.xlabel('x [mm]')
        plt.ylabel('Intensity')
        plt.title(name)
        plt.legend(loc='upper left',fontsize=17)
        plt.tight_layout()
        plt.savefig(os.path.join(figpath,name+'average_x_both'))

        figbothy = plt.figure()

        # G4beamline data
        plt.plot(newbins,yn,'o',color='green',marker='^',label='model')

        # Real data
        plt.scatter(xvals,ydataavg,color='blue',label='data')
        plt.xlabel('y [mm]')
        plt.ylabel('Intensity')
#     plt.legend(loc='upper right')
        plt.title(name)
        plt.legend(loc='upper left',fontsize=17)
        plt.tight_layout()
        plt.savefig(os.path.join(figpath,name+'average_y_both'))
    


    plt.close('all')


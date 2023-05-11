# various ploting and analysis tools for analysis of M4 beam line on 
# fermilab's Muon campus
# part of the SULI 2022 summer research project
# Author: Trevor Loe

from tkinter import N, Y
from trace import CoverageResults
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import shutil
from scipy.optimize import curve_fit
import scipy.optimize as opt
from gekko import GEKKO
import random
from matplotlib.patches import Ellipse
import matplotlib


def read_data(filename):
    # opens the file given by the string 'filename' and returns a pandas dataframe of values in the file
    print('loading ASCII file: '+filename,end='\n')

    f = open(filename, 'r')
    #check number of line
    all_lines = f.readlines()
    N = len(all_lines)

    # create numpy array
    # there are 12 pieces of data in each row
    npdata = np.zeros([N-2,28])         # subtract 2 off of N to account for first two rows of info

    
    #print(all_lines)
    for i,line in enumerate(all_lines):
        print(i,end='\r')
        # print(line)
        if i<2:
            pass
        else:
            nums = line.split(' ')
            # print(nums)
            # print(len(nums))
            for j,num in enumerate(nums):
                if j>11:
                    pass
                else:
                    # print(num)
                    npdata[i-2,j] = float(num)

    col_lab = all_lines[1].split(' ')
    # print(col_lab)
    # print(col_lab)
    df = pd.DataFrame(npdata,columns = col_lab)

    return df


def read_scan_beamdata(filename):
    # loads data from csv containing actual M4beamline data taken
    print('loading csv data: ')
    print(filename)

    df = pd.read_csv(filename,sep=';')

    # print(df)
    # print(df['mw_reading_0_95'])

    return(df)


def make_scan_beam_hist(data,samplename):
    # creates histogram from selected sample in data
    # creates the histogram in the currently active figure, must have figure created to do
    # must run plt.show() to show figure after
    index = data[data['data_label']==samplename].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_1':'mw_reading_0_95']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()

    plt.hist(col,bins=15)


def make_scan_beam_scatterplot(data,samplename):
    # creates a scatter plot for the selected sample in data
    # creates lineplot in the currently active figure, must have figure created
    # must run plt.show() after
    index = data[data['data_label']==samplename].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_1':'mw_reading_0_95']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()

    plt.scatter(np.linspace(0,len(col),len(col)),col)

def make_mw_beam_hist(data,samplename):
    # creates histogram from selected sample in data
    # creates the histogram in the currently active figure, must have figure created to do
    # must run plt.show() to show figure after
    index = data[data['data_label_0']==samplename].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_1':'mw_reading_0_95']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()

    plt.hist(col,bins=15)


def make_mw_beam_scatterplot(data,samplename):
    # creates a scatter plot for the selected sample in data
    # creates lineplot in the currently active figure, must have figure created
    # must run plt.show() after
    index = data[data['data_label_0']==samplename].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_1':'mw_reading_0_95']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()

    plt.scatter(np.linspace(0,len(col),len(col)),col)


def make_mw_beam_bar(data,samplename):
    # creates a bar plot for the selected sample in data
    # must have a figure and axes for the bar chart to appear on
    # must run plt.show() after
    index = data[data['data_label_0']==samplename].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_1':'mw_reading_0_95']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()
    height = col.to_numpy()
    height = np.array(height)
    height=height[:,0]
    plt.bar(np.linspace(0,len(height),len(height)),height)


def make_scan_beam_bar(data,samplename):
    # creates a bar plot for the selected sample in data
    # must have a figure and axes for the bar chart to appear on
    # must run plt.show() after
    index = data[data['data_label']==samplename].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_1':'mw_reading_0_95']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()
    height = col.to_numpy()
    height = np.array(height)
    height=height[:,0]
    plt.bar(np.linspace(0,len(height),len(height)),height)



def save_3hist(data, name):
    # save the figures for the 3 histograms created by the x,y, and z data given in 'data'

    bins = 40
    xlimits = (-15,15)

    # x data
    figx = plt.figure()
    plt.hist(data['#x'], bins=bins)
    # figure labels and stuff
    plt.xlabel('x [mm]')
    plt.ylabel('Events')
    plt.xlim(xlimits)

    plt.savefig('histograms/x'+name)

    print('x std: ')
    print(data['#x'].std())

    # y data
    figy = plt.figure()
    plt.hist(data['y'], bins=bins)
    # figure labels and stuff
    plt.xlabel('y [mm]')
    plt.ylabel('Events')
    plt.xlim(xlimits)

    plt.savefig('histograms/y'+name)

    # z data
    figz = plt.figure()
    plt.hist(data['z'], bins=bins)
    # figure labels and stuff
    plt.xlabel('z [mm]')
    plt.ylabel('Events')
    plt.xlim(xlimits)

    plt.savefig('histograms/z'+name)

    plt.close('all')


def make_hist_dir(folder):
    # makes a folder for saving several histograms
    # N O T E: will delete the directory with the same name if it exists
    
    # make folder
    curr = os.getcwd()
    temp = os.path.join(curr,'data_histograms/')
    foldpath = os.path.join(temp,folder)

    try:
        os.mkdir(foldpath)
    except FileExistsError:
        print('Directory exists, deleting...')
        shutil.rmtree(foldpath)
        os.mkdir(foldpath)


def data_std(values):
    # finds the standard deviation of a set of data specfied by 'values'
    # values is given as a numpy array (should  be 48 values)

    # first create integer roundings for each value in values
    intvals = np.round(values*1e5)
    # print(intvals)
    # then create numpy arrays for each value
    indv_arrays = []
    pos = np.linspace(-24,24,len(values))
    for idx,num in enumerate(intvals):
        array = pos[idx]*np.ones((1,int(num)))

        indv_arrays.append(array.T)
    

    # print(indv_arrays)
    full_data = np.concatenate(indv_arrays)
    
    # calculate the standard deviation of this data set
    return np.std(full_data)


def shift_data(data,num):
    # shifts over the values in the array 'data' but an amount, 'num' to the RIGHT
    # data points that need to be added to an end just become zero
    newdata = np.zeros(data.shape)

    for i,val in enumerate(data):
        if i+num<0 or i+num>=len(data):
            continue
        else:
            newdata[i+num] = val

    return newdata


def save_2bar(data,samplename,det,folder=None):
    # saves the histogram for the x and y data given by the data frame 'data'
    # samplename specifies what row the data is one
    # det specfies what detector the data corresponds to
    # folder is the name of the folder to which they will be saved in data_historgrams directory (currently only uses for printing messages)
    curr = os.getcwd()
    temp = os.path.join(curr,'data_histograms/')
    foldpath = os.path.join(temp,folder)

    # y data is the first 0-47, x data is the 48-95 mw readings (subject to change)

    figx = plt.figure()

    # create y plot
    index = data[data['data_label_0']==samplename].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_0':'mw_reading_0_47']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()
    height = col.to_numpy()
    height = np.array(height)
    height=-height[:,0]
    # make the minimum value in the array 0 (bring up all values so they are greater than 0)
    height = height - np.min(height)

    # zero out values below 0.1 of the maximum (for visualization purposes)
    # height[height<0.1*np.max(height)]=0

    # get mean and std
    # normheight = height/np.max(height)
    yvals = np.linspace(-24,24,len(height))
    ymean = np.average(yvals,weights=height)
    # yvarience = np.average((yvals-ymean)**2, weights=normheight)
    # # print(yvarience)
    # # xstd = math.sqrt(xvarience)
    
    # # print(normheight)
    # yvarience = np.sum(normheight*(yvals-ymean)**2)/len(yvals)
    # # print(yvarience)

    # fw = np.round(normheight*1e7)
    # # print(fw)
    # yvarience = np.cov(yvals, aweights=normheight)
    # print(yvarience)

    # ystd = math.sqrt(yvarience)
    # ystd = data_std(height)

    # normalize intensities to be probabilities
    normheight = height/np.sum(height)
    # (test) zero out data below a threshold
    normheight[height<0.1*np.max(height)]=0
    yvarience = np.sum(((yvals-ymean)**2)*normheight)
    ystd = math.sqrt(yvarience)

    ydist = round(ymean/(48/len(height)))
    # print(ydist)
    nheight = shift_data(height,-ydist)

    plt.bar(np.linspace(-24,24,len(nheight)),nheight)

    plt.xlabel('y [mm]')
    plt.ylabel('Intensity')
    plt.title(det)

    plt.savefig('data_histograms/'+folder+'/y'+det)

    # print(ystd)

    figy = plt.figure()

    # create x plot
    index = data[data['data_label_0']==samplename].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_48':'mw_reading_0_95']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()
    height = col.to_numpy()
    height = np.array(height)
    height=-height[:,0]
    # make the minimum value in the array 0 (bring up all values so they are greater than 0)
    height = height - np.min(height)

    # zero out values below 0.1 of the maximum (for visualization purposes)
    # height[height<0.1*np.max(height)]=0

    # get mean and std
    # normheight = height/np.max(height)
    xvals = np.linspace(-24,24,len(height))
    xmean = np.average(xvals,weights=height)
    # xvarience = np.average((xvals-xmean)**2, weights=normheight)
    # # print(xvarience)
    # # xstd = math.sqrt(xvarience)
    
    # # print(normheight)
    # xvarience = np.sum(normheight*(xvals-xmean)**2)/len(xvals)
    # # print(xvarience)

    # fw = np.round(normheight*1e7)
    # # print(fw)
    # xvarience = np.cov(xvals, aweights=normheight)
    # print(xvarience)
    # xstd = math.sqrt(xvarience)
    # xstd = data_std(height)
    # print(xstd)

    # normalize intensities to be probabilities 
    normheight = height/(np.sum(height))
    # (test) zero out data below a threshold
    normheight[height<0.1*np.max(height)]=0
    xvarience = np.sum(((xvals-xmean)**2)*normheight)
    xstd = math.sqrt(xvarience)

    # get distance (in bins) between the mean and 0
    xdist = round(xmean/(48/len(height)))
    # print(xdist)
    nheight = shift_data(height,-xdist)

    plt.bar(np.linspace(-24,24,len(nheight)),nheight)

    plt.xlabel('x [mm]')
    plt.ylabel('Intensity')
    plt.title(det)

    plt.savefig('data_histograms/'+folder+'/x'+det)
    plt.close('all')

    print('x and y figured saved in '+foldpath)

    return (xmean,xstd,ymean,ystd)

def extract_rowx(data,rowname):
    # pull out one row from the datafarme 'data' and return it as a numpy array

    # create y plot
    index = data[data['data_label_0']==rowname].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_0':'mw_reading_0_47']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()
    height = col.to_numpy()
    height = np.array(height)
    height=height[:,0]

    return height

def extract_rowy(data,rowname):
    # pull out one row from the datafarme 'data' and return it as a numpy array

    # create y plot
    index = data[data['data_label_0']==rowname].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_48':'mw_reading_0_95']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()
    height = col.to_numpy()
    height = np.array(height)
    height=height[:,0]

    return height

def extract_rowx_scan(data,rowname):
    # pull out one row from the datafarme 'data' and return it as a numpy array

    # create y plot
    index = data[data['data_label']==rowname].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_0':'mw_reading_0_47']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()
    height = col.to_numpy()
    height = np.array(height)
    height=height[:,0]

    return height

def extract_rowy_scan(data,rowname):
    # pull out one row from the datafarme 'data' and return it as a numpy array

    # create y plot
    index = data[data['data_label']==rowname].index.values[0]

    row = test_row = data.loc[[index],'mw_reading_0_48':'mw_reading_0_95']
    # row = test_row = data.loc[[index],'blm_reading_1':'blm_reading_51']
    row_fl = row.astype(float)
    col = row_fl.transpose()
    height = col.to_numpy()
    height = np.array(height)
    height=height[:,0]

    return height

def cutoff_noise(data,xdata):
    # takes the ydata and xdata from a detector and outputs another set of data without 'noise' values
    # noise values are taken to be values below a certain threshold
    
    boolarray = data<0.1*np.min(data)
    # print(boolarray)
    idx = [i for i, x in enumerate(boolarray) if x]

    # print(idx)
    newdata = data[idx]
    newx = xdata[idx]
    # fig = plt.figure()
    # plt.plot(newx,newdata,'o')
    # plt.title('cutoff data')
    return newdata,newx



def guass(x,A,x0,sig):
    y = A*np.exp(-(x-x0)**2/(2*sig**2))
    return y

def fit_guass(data):
    # takes data as a numpy array and returns guassian fit parameters
    # fits to data that has been raised up so that the smalleest value is zero

    # data will be all negative (mostly)
    data = data - np.max(data)
    # data[data>0.1*np.min(data)]=0.1*np.min(data)
    # data = data - np.max(data)
    xdata = np.linspace(-24,24,len(data))
    data,xdata = cutoff_noise(data,xdata)

    # print(data)
    
    parameters, covarience = curve_fit(guass,xdata,data,p0=(-1,-5,3))

    A = parameters[0]
    x0 = parameters[1]
    sig = parameters[2]

    # print(covarience)
    err = np.sqrt(np.diag(covarience))
    return (A,x0,sig,err)

def fit_plot_guass(data,name):
    # fits and plots a guassian to the numpy array 'data'
    # must have matplotlib figure initialized

    A,x0,sig,err = fit_guass(data)

    # print(A)
    # print(x0)
    # print(sig)
    # print(err)

    data = data - np.max(data)
    # data[data>0.1*np.min(data)]=0.1*np.min(data)
    # data = data - np.max(data)

    xdata = np.linspace(-24,24,len(data))
    fitdata = guass(xdata,A,x0,sig)

    plt.plot(xdata,data,'o',label='data')
    plt.plot(xdata,fitdata,'-',label='guassian fit')
    plt.legend()
    plt.title('Gaussian fit for '+name)

    return (sig,x0,err)

def thrsc(sig1,sig2,sig3,d1,d2,d3):
    # takes 3 sets of x data and uses the 3-screen method to compute beta_0, alpha_0, and eps
    # baseline equation for each dataset:
    # sig^2 = beta_0*eps - 2d*alpha_0*eps + d^2*eps*(1+alpha_0^2)/beta_0
    # data given as numpy arrays (data not used in this implementation)

    # sig1 = np.std(data1)
    # sig2 = np.std(data2)
    # sig3 = np.std(data3)

    def sys(p,*args):
        # system for 3 coupled equations whos solutions yield the twist parameters
        alpha,beta,eps = p

        sig1 = args[0]
        sig2 = args[1]
        sig3 = args[2]
        d1 = args[3]
        d2 = args[4]
        d3 = args[5]

        e1 = beta*eps - 2*d1*alpha*eps + d1**2*eps*(1+alpha**2)/beta - sig1**2
        e2 = beta*eps - 2*d2*alpha*eps + d2**2*eps*(1+alpha**2)/beta - sig2**2
        e3 = beta*eps - 2*d3*alpha*eps + d3**2*eps*(1+alpha**2)/beta - sig3**2

        return (e1,e2,e3)

    args = (sig1,sig2,sig3,d1,d2,d3)

    # print(args)
    # (alpha,beta,eps),infodict,ier,mesg = opt.fsolve(sys,(0.19,4.01,2.4e-7),args=args,full_output=1)
    # (alpha,beta,eps),infodict,ier,mesg = opt.fsolve(sys,(1,20,2e-7),args=args,full_output=1)
    (alpha,beta,eps),infodict,ier,mesg = opt.fsolve(sys,(1,7,5e-7),args=args,full_output=1)

    # print(infodict)
    # print(mesg)
    # print(sys((alpha,beta,eps),*args))
    print('converged? '+str(ier))

    return alpha,beta,eps,ier

def thrsc_gek(sig1,sig2,sig3,d1,d2,d3):
    # takes 3 sets of x data and uses the 3-screen method to compute beta_0, alpha_0, and eps
    # baseline equation for each dataset:
    # sig^2 = beta_0*eps - 2d*alpha_0*eps + d^2*eps*(1+alpha_0^2)/beta_0
    # data given as numpy arrays
    # uses GEKKO implementation rather than fsolve

    # sig1 = np.std(data1)
    # sig2 = np.std(data2)
    # sig3 = np.std(data3)

    m = GEKKO()
    # alpha = m.Var(value=-1)
    # beta = m.Var(value=20)
    # eps = m.Var(value=2e-7)
    alpha = m.Var(value=1)
    beta = m.Var(value=20)
    eps = m.Var(value=5e-7)

    m.Equations([beta*eps - 2*d1*alpha*eps + d1**2*eps*(1+alpha**2)/beta==sig1**2,\
                beta*eps - 2*d2*alpha*eps + d2**2*eps*(1+alpha**2)/beta==sig2**2,\
                beta*eps - 2*d3*alpha*eps + d3**2*eps*(1+alpha**2)/beta==sig3**2])

    # m.options.SOLVER = 3
    # m.options.IMODE = 5
    # m.solver_options = ['minlp_gap_tol 0.01']
    m.options.LINEAR = 0
    m.options.MAX_ITER = 400
    m.options.OTOL = 1e-8
    m.options.RTOL = 1e-8
    m.solve(disp=False)
    bval = beta.value[0]
    aval = alpha.value[0]
    eval = eps.value[0]
    # print(bval*eval - 2*d1*aval*eval + d1**2*eval*(1+aval**2)/bval)
    # print(sig1**2)

    return alpha.value,beta.value,eps.value
    
def random_sig_solve(sig1,sig2,sig3,err1,err2,err3,d1,d2,d3):
    # attempts to solve the 3 equation nonlinear system for the 3 screen method by picking
    # a random sigma for each sigma within the err range specified by err

    # print(sig1)
    # print(sig2)
    # print(sig3)
    newsig1 = random.uniform(sig1-1.5*abs(err1),sig1+1.5*abs(err1))
    newsig2 = random.uniform(sig2-1.5*abs(err2),sig2+1.5*abs(err2))
    newsig3 = random.uniform(sig3-1.5*abs(err3),sig3+1.5*abs(err3))

    # print(newsig1)
    # print(newsig2)
    # print(newsig3)

    # first use scipy
    alpha1,beta1,eps1,ier = thrsc(newsig1/1000,newsig2/1000,newsig3/1000,d1,d2,d3)

    if ier==1:
        sci_conv = 1
    else:
        sci_conv = 0

    # use gekko
    try:
        alpha2,beta2,eps2 = thrsc_gek(sig1/1000,sig2/1000,sig3/1000,d1,d2,d3)
        gek_conv = 1
    except Exception:
        gek_conv = 0
        alpha2=[float('nan')]
        beta2=[float('nan')]
        eps2=[float('nan')]

    return (alpha1,beta1,eps1,sci_conv,alpha2[0],beta2[0],eps2[0],gek_conv,newsig1,newsig2,newsig3)


    # define fitclass so that I can pass parameters into my fitting fucntion
class fitClass:

    def __init__(self):
        pass

    def fitfunc(self, x, A, B, C):
        y = A*x**2 - 2*B*self.d*x + C*self.d**2
        return y

def fit_plot_parabola(xdata,ydata,yuncert,d,xflag):
    # fits a parabola to the given data and extracts the twist parameters from that parabola fit
    # also creates of plot of the fit along with the data
    # must have figure initialized already
    # xflag is true if the data is for x and false if the data if for y

    # create fit class
    inst = fitClass()
    inst.d = d

    # get rid of nans in data
    xdata,ydata,yuncert = remove_nan(xdata,ydata,yuncert)

    # print(yuncert)
    coef, cov = curve_fit(inst.fitfunc,xdata,ydata,sigma=yuncert)
    A = coef[0]
    B = coef[1]
    C = coef[2]
    uncert = np.sqrt(np.diag(cov))
    dA = uncert[0]
    dB = uncert[1]
    dC = uncert[2]

    # create plot
    plt.errorbar(xdata,ydata,yerr=yuncert,marker='o',linestyle='',label='data')

    # plot fitted parabola
    xth = np.linspace(xdata[0],xdata[-1],200)
    plt.plot(xth,inst.fitfunc(xth,A,B,C),color='g',label='fitted parabola')
    plt.xlabel('$1 - d/f$')
    if xflag:
        plt.ylabel('$\\langle x \\rangle^2$')
        plt.xlabel('$1 + d/f$')
    else:
        plt.ylabel('$\\langle y \\rangle^2$')
        plt.xlabel('$1 - d/f$')
    plt.legend()

    # print(A)
    # print(B)
    # print(C)
    plt.show()
    # then extract twist parameters from fit parameters
    eps = math.sqrt(A*C - B**2)
    beta = A/eps
    alpha = B/eps
    gamma = C/eps

    # propogate uncertainties
    depsdA = C*1/(2*math.sqrt(A*C-B**2))
    depsdC = A*1/(2*math.sqrt(A*C-B**2))
    depsdB = 1/(2*math.sqrt(A*C-B**2))

    eps_uncert = math.sqrt((depsdA*dA)**2 + (depsdC*dC)**2 + (depsdB*dB)**2)

    beta_uncert = abs(beta)*math.sqrt((eps_uncert/eps)**2+(dA/A)**2)
    alpha_uncert = abs(alpha)*math.sqrt((eps_uncert/eps)**2+(dB/B)**2)
    gamma_uncert = abs(gamma)*math.sqrt((eps_uncert/eps)**2+(dC/C)**2)
    # (alpha,beta,gamma,eps,alpha_uncert,beta_uncert,gamma_uncert,eps_uncert) = (0,0,0,0,0,0,0,0)

    return (alpha,beta,gamma,eps,alpha_uncert,beta_uncert,gamma_uncert,eps_uncert)

def remove_nan(xdata,ydata,yerr):
    # returns data without the elements in ydata that were nan
    
    newx = []
    newy = []
    newyerr = []

    for ii,y in enumerate(ydata):
        if math.isnan(y):
            continue
        else:
            newy.append(y)
            newx.append(xdata[ii])
            newyerr.append(yerr[ii])

    return (np.array(newx),np.array(newy),np.array(newyerr))


def new_guass_fit(xdata,ydata,oxdata,oydata,name,xflag):
    # fits and plots a guassian fit of the data GIVEN (not necessarily all 48 data points)
    # needs figure initialized

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 11}

    matplotlib.rc('font', **font)

    parameters, covarience = curve_fit(guass,xdata,ydata,p0=(-1,-5,3))

    # oxdata = np.linspace(-24,24,48)
    xth = np.linspace(-24,24,200)
    fitdata = guass(xth,parameters[0],parameters[1],parameters[2])

    plt.plot(oxdata,oydata,'o',color='blue',label='Original data')
    plt.plot(xdata,ydata,'o',color='red',label='Data used for fit')
    plt.plot(xth,fitdata,'-',color='orange',label='Gaussian fit')

    if xflag:
        plt.xlabel('x [mm]')
    else:
        plt.xlabel('y [mm]')

    plt.ylabel('Intensity')
    plt.legend(loc='lower left')
    plt.title(name)
    plt.tight_layout()

    err = np.sqrt(np.diag(covarience))
    return parameters[2],err[2]

def guass_fit_wcutoff(xdata,ydata,lcut,rcut,offset,name,xflag):
    # cuts off and offsets data from x and y array and then called new_guass_fit
    # needs figure initialized

    # first offset
    ydata = ydata + offset
    # slice off first lcut points
    newydata = ydata[lcut:-1]
    newxdata = xdata[lcut:-1]
    # slice off last rcut points
    newydata = newydata[0:-rcut]
    newxdata = newxdata[0:-rcut]

    # print(newxdata)
    # print(newydata)

    sig,err = new_guass_fit(newxdata,newydata,xdata,ydata,name,xflag)

    return (sig,err)


def comp_3sc_uncert(sig1,sig2,sig3,err1,err2,err3,d1,d2,d3):
    # uses the 3 sigmas given and their uncertainties to compute the range of twist parameters possible within those uncertainties

    # get baseline twist params for comparison
    alphab,betab,epsb,ier = thrsc(sig1,sig2,sig3,d1,d2,d3)

    # 2^3 = 8 compinations
    # low high high - 1
    # low low high - 2
    # low low low - 3
    # low high low - 4
    # high low low - 5
    # high low high - 6
    # high high low - 7
    # high high high - 8

    low1 = abs(sig1) - abs(err1)
    low2 = abs(sig2) - abs(err2)
    low3 = abs(sig3) - abs(err3)
    hi1 = abs(sig1) + abs(err1)
    hi2 = abs(sig2) + abs(err2)
    hi3 = abs(sig3) + abs(err3)

    all_alpha = np.zeros([2,8])
    all_beta = np.zeros([2,8])
    all_eps = np.zeros([2,8])
    # print(all_alpha[1,1])
    # print(sig1)
    # print(sig2)
    # print(sig3)
    # print(err1)
    # print(err2)
    # print(err3)
    # case 1
    alpha1,beta1,eps1,ier = thrsc(low1,hi2,hi3,d1,d2,d3)
    if ier!=1:
        print('scipy didnt converge')
        alpha1=float('nan')
        beta1=float('nan')
        eps1=float('nan')
    
    try:
        alpha2,beta2,eps2 = thrsc_gek(low1,hi2,hi3,d1,d2,d3)
    except Exception:
        print('gekko did not converge')
        alpha2=[float('nan')]
        beta2=[float('nan')]
        eps2=[float('nan')]

    alpha2 = alpha2[0]
    beta2 = beta2[0]
    eps2 = eps2[0]
    # fix errors of different alogrithm getting different signs for the parameters
    if (alpha1>0 and alpha2<0) or (alpha1<0 and alpha2>0):
        alpha2 = -alpha2
    if (beta1>0 and beta2<0) or (beta1<0 and beta2>0):
        beta2 = -beta2
    if (eps1>0 and eps2<0) or (eps1<0 and eps2>0):
        eps2 = -eps2
    
    # all_alpha[:,1] = np.array([[alpha1],[alpha2]])
    all_alpha[:,0] = [alpha1,alpha2]
    all_beta[:,0] = np.array([beta1,beta2])
    all_eps[:,0] = np.array([eps1,eps2])

    # case 2
    alpha1,beta1,eps1,ier = thrsc(low1,low2,hi3,d1,d2,d3)
    if ier!=1:
        print('scipy didnt converge')
        alpha1=float('nan')
        beta1=float('nan')
        eps1=float('nan')
    
    try:
        alpha2,beta2,eps2 = thrsc_gek(low1,low2,hi3,d1,d2,d3)
    except Exception:
        print('gekko did not converge')
        alpha2=[float('nan')]
        beta2=[float('nan')]
        eps2=[float('nan')]

    alpha2 = alpha2[0]
    beta2 = beta2[0]
    eps2 = eps2[0]
    # fix errors of different alogrithm getting different signs for the parameters
    if (alpha1>0 and alpha2<0) or (alpha1<0 and alpha2>0):
        alpha2 = -alpha2
    if (beta1>0 and beta2<0) or (beta1<0 and beta2>0):
        beta2 = -beta2
    if (eps1>0 and eps2<0) or (eps1<0 and eps2>0):
        eps2 = -eps2
    
    all_alpha[:,1] = [alpha1,alpha2]
    all_beta[:,1] = [beta1,beta2]
    all_eps[:,1] = [eps1,eps2]

    # case 3
    alpha1,beta1,eps1,ier = thrsc(low1,low2,low3,d1,d2,d3)
    if ier!=1:
        print('scipy didnt converge')
        alpha1=float('nan')
        beta1=float('nan')
        eps1=float('nan')
    
    try:
        alpha2,beta2,eps2 = thrsc_gek(low1,low2,low3,d1,d2,d3)
    except Exception:
        print('gekko did not converge')
        alpha2=[float('nan')]
        beta2=[float('nan')]
        eps2=[float('nan')]

    alpha2 = alpha2[0]
    beta2 = beta2[0]
    eps2 = eps2[0]
    # fix errors of different alogrithm getting different signs for the parameters
    if (alpha1>0 and alpha2<0) or (alpha1<0 and alpha2>0):
        alpha2 = -alpha2
    if (beta1>0 and beta2<0) or (beta1<0 and beta2>0):
        beta2 = -beta2
    if (eps1>0 and eps2<0) or (eps1<0 and eps2>0):
        eps2 = -eps2
    
    all_alpha[:,2] = [alpha1,alpha2]
    all_beta[:,2] = [beta1,beta2]
    all_eps[:,2] = [eps1,eps2]

    # case 4
    alpha1,beta1,eps1,ier = thrsc(low1,hi2,low3,d1,d2,d3)
    if ier!=1:
        print('scipy didnt converge')
        alpha1=float('nan')
        beta1=float('nan')
        eps1=float('nan')
    
    try:
        alpha2,beta2,eps2 = thrsc_gek(low1,hi2,low3,d1,d2,d3)
    except Exception:
        print('gekko did not converge')
        alpha2=[float('nan')]
        beta2=[float('nan')]
        eps2=[float('nan')]

    alpha2 = alpha2[0]
    beta2 = beta2[0]
    eps2 = eps2[0]
    # fix errors of different alogrithm getting different signs for the parameters
    if (alpha1>0 and alpha2<0) or (alpha1<0 and alpha2>0):
        alpha2 = -alpha2
    if (beta1>0 and beta2<0) or (beta1<0 and beta2>0):
        beta2 = -beta2
    if (eps1>0 and eps2<0) or (eps1<0 and eps2>0):
        eps2 = -eps2
    
    all_alpha[:,3] = [alpha1,alpha2]
    all_beta[:,3] = [beta1,beta2]
    all_eps[:,3] = [eps1,eps2]

    # case 5
    alpha1,beta1,eps1,ier = thrsc(hi1,low2,low3,d1,d2,d3)
    if ier!=1:
        print('scipy didnt converge')
        alpha1=float('nan')
        beta1=float('nan')
        eps1=float('nan')
    
    try:
        alpha2,beta2,eps2 = thrsc_gek(hi1,low2,low3,d1,d2,d3)
    except Exception:
        print('gekko did not converge')
        alpha2=[float('nan')]
        beta2=[float('nan')]
        eps2=[float('nan')]

    alpha2 = alpha2[0]
    beta2 = beta2[0]
    eps2 = eps2[0]
    # fix errors of different alogrithm getting different signs for the parameters
    if (alpha1>0 and alpha2<0) or (alpha1<0 and alpha2>0):
        alpha2 = -alpha2
    if (beta1>0 and beta2<0) or (beta1<0 and beta2>0):
        beta2 = -beta2
    if (eps1>0 and eps2<0) or (eps1<0 and eps2>0):
        eps2 = -eps2
    
    all_alpha[:,4] = [alpha1,alpha2]
    all_beta[:,4] = [beta1,beta2]
    all_eps[:,4] = [eps1,eps2]

    # case 6
    alpha1,beta1,eps1,ier = thrsc(hi1,low2,hi3,d1,d2,d3)
    if ier!=1:
        print('scipy didnt converge')
        alpha1=float('nan')
        beta1=float('nan')
        eps1=float('nan')
    
    try:
        alpha2,beta2,eps2 = thrsc_gek(hi1,low2,hi3,d1,d2,d3)
    except Exception:
        print('gekko did not converge')
        alpha2=[float('nan')]
        beta2=[float('nan')]
        eps2=[float('nan')]

    alpha2 = alpha2[0]
    beta2 = beta2[0]
    eps2 = eps2[0]
    # fix errors of different alogrithm getting different signs for the parameters
    if (alpha1>0 and alpha2<0) or (alpha1<0 and alpha2>0):
        alpha2 = -alpha2
    if (beta1>0 and beta2<0) or (beta1<0 and beta2>0):
        beta2 = -beta2
    if (eps1>0 and eps2<0) or (eps1<0 and eps2>0):
        eps2 = -eps2
    
    all_alpha[:,5] = [alpha1,alpha2]
    all_beta[:,5] = [beta1,beta2]
    all_eps[:,5] = [eps1,eps2]

    # case 7
    alpha1,beta1,eps1,ier = thrsc(hi1,hi2,low3,d1,d2,d3)
    if ier!=1:
        print('scipy didnt converge')
        alpha1=float('nan')
        beta1=float('nan')
        eps1=float('nan')
    
    try:
        alpha2,beta2,eps2 = thrsc_gek(hi1,hi2,low3,d1,d2,d3)
    except Exception:
        print('gekko did not converge')
        alpha2=[float('nan')]
        beta2=[float('nan')]
        eps2=[float('nan')]

    alpha2 = alpha2[0]
    beta2 = beta2[0]
    eps2 = eps2[0]
    # fix errors of different alogrithm getting different signs for the parameters
    if (alpha1>0 and alpha2<0) or (alpha1<0 and alpha2>0):
        alpha2 = -alpha2
    if (beta1>0 and beta2<0) or (beta1<0 and beta2>0):
        beta2 = -beta2
    if (eps1>0 and eps2<0) or (eps1<0 and eps2>0):
        eps2 = -eps2
    
    all_alpha[:,6] = [alpha1,alpha2]
    all_beta[:,6] = [beta1,beta2]
    all_eps[:,6] = [eps1,eps2]

    # case 8
    alpha1,beta1,eps1,ier = thrsc(hi1,hi2,hi3,d1,d2,d3)
    if ier!=1:
        print('scipy didnt converge')
        alpha1=float('nan')
        beta1=float('nan')
        eps1=float('nan')
    
    try:
        alpha2,beta2,eps2 = thrsc_gek(hi1,hi2,hi3,d1,d2,d3)
    except Exception:
        print('gekko did not converge')
        alpha2=[float('nan')]
        beta2=[float('nan')]
        eps2=[float('nan')]

    alpha2 = alpha2[0]
    beta2 = beta2[0]
    eps2 = eps2[0]
    # fix errors of different alogrithm getting different signs for the parameters
    if (alpha1>0 and alpha2<0) or (alpha1<0 and alpha2>0):
        alpha2 = -alpha2
    if (beta1>0 and beta2<0) or (beta1<0 and beta2>0):
        beta2 = -beta2
    if (eps1>0 and eps2<0) or (eps1<0 and eps2>0):
        eps2 = -eps2
    
    all_alpha[:,7] = [alpha1,alpha2]
    all_beta[:,7] = [beta1,beta2]
    all_eps[:,7] = [eps1,eps2]


    print(all_alpha)
    print(all_beta)
    print(all_eps)
    print(alphab)
    print(betab)
    print(epsb)
    # compute max deviation
    alpha_uncert = np.nanmax(np.abs(all_alpha-alphab))
    beta_uncert = np.nanmax(np.abs(all_beta-betab))
    eps_uncert = np.nanmax(np.abs(all_eps-epsb))

    return alpha_uncert,beta_uncert,eps_uncert

def new_quad_scan925(foc,sigsqr,uncert):
    # new quad scan function
    # must have figure initialized
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17}

    matplotlib.rc('font', **font)

    foc,sigsqr,uncert = remove_nan(foc,sigsqr,uncert)

    def parab(x,A,B,C):
         d = 33.41371 

         y = A*(x**2) - 2*d*B*x + C*(d**2)

         return y

    coef, cov = curve_fit(parab,foc,sigsqr,sigma=uncert)

    A = coef[0]
    B = coef[1]
    C = coef[2]

    plt.errorbar(foc,sigsqr*1e6,uncert*1e6,marker='o',capsize=4,color='green',label='Data',linestyle='')

    xth = np.linspace(np.min(foc),np.max(foc),200)
    fitted_data = parab(xth,A,B,C)
    plt.plot(xth,fitted_data*1e6,color='red',label='Parabola fit')
    plt.xlabel('$1-\\frac{d}{f}$')
    plt.ylabel('$\\sigma_y^2$ [mm$^2$]')
    plt.legend()
    plt.tight_layout()

    # plt.show()
    # print(A)
    # print(B)
    # print(C)
    # print(A*C - B**2)
    err = np.sqrt(np.diag(cov))
    dA = err[0]
    dB = err[1]
    dC = err[2]

    eps = math.sqrt(A*C - B**2)
    beta = A/eps
    alpha = B/eps
    gamma = C/eps

    # print(alpha)
    # print(beta)
    # print(eps)
    # print(gamma)

    # uncertainties
    depsdA = C/(2*eps)
    depsdB = -B/eps
    depsdC = A/(2*eps)
    deps = math.sqrt((depsdA*dA)**2 + (depsdB*dB)**2 + (depsdC*dC)**2)

    dbeta = abs(beta)*math.sqrt((dA/A)**2+(deps/eps)**2)
    dalpha = abs(alpha)*math.sqrt((dB/B)**2 + (deps/eps)**2)
    dgamma = abs(gamma)*math.sqrt((dC/C)**2 + (deps/eps)**2)


    return (alpha,beta,gamma,eps,dalpha,dbeta,dgamma,deps)


def new_quad_scan930(foc,sigsqr,uncert):
    # new quad scan function
    # must have figure initialized

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17}

    matplotlib.rc('font', **font)

    foc,sigsqr,uncert = remove_nan(foc,sigsqr,uncert)

    def parab(x,A,B,C):
         d = 25.14248

         y = A*(x**2) - 2*d*B*x + C*(d**2)

         return y

    coef, cov = curve_fit(parab,foc,sigsqr,sigma=uncert)

    A = coef[0]
    B = coef[1]
    C = coef[2]

    plt.errorbar(foc,sigsqr*1e6,uncert*1e6,marker='o',capsize=4,color='green',label='Data',linestyle='')

    xth = np.linspace(np.min(foc),np.max(foc),200)
    fitted_data = parab(xth,A,B,C)
    plt.plot(xth,fitted_data*1e6,color='red',label='Parabola fit')
    plt.xlabel('$1-\\frac{d}{f}$')
    plt.ylabel('$\\sigma_x^2$ [mm$^2$]')
    plt.legend()
    plt.tight_layout()

    # plt.show()
    # print(A)
    # print(B)
    # print(C)
    # print(A*C - B**2)
    err = np.sqrt(np.diag(cov))
    dA = err[0]
    dB = err[1]
    dC = err[2]

    eps = math.sqrt(A*C - B**2)
    beta = A/eps
    alpha = B/eps
    gamma = C/eps

    # print(alpha)
    # print(beta)
    # print(eps)
    # print(gamma)

    # uncertainties
    # uncertainties
    depsdA = C/(2*eps)
    depsdB = -B/eps
    depsdC = A/(2*eps)
    deps = math.sqrt((depsdA*dA)**2 + (depsdB*dB)**2 + (depsdC*dC)**2)

    dbeta = abs(beta)*math.sqrt((dA/A)**2+(deps/eps)**2)
    dalpha = abs(alpha)*math.sqrt((dB/B)**2 + (deps/eps)**2)
    dgamma = abs(gamma)*math.sqrt((dC/C)**2 + (deps/eps)**2)

    return (alpha,beta,gamma,eps,dalpha,dbeta,dgamma,deps)

def make_ellipse(alpha,beta,emit,linstyle,color,label):
    # make and plot an ellipse from the given twist parameters
    # must have figure initialized
    # plots it in mm

    # # make a large grid of points, keep the ones that satisfy the ellipse equation
    # x = np.linspace(-1e-4,1e-4,500)
    # y = np.linspace(-1e-4,1e-4,500)

    # (X,Y) = np.meshgrid(x,y)

    # gamma = (1+alpha**2)/beta
    # width = math.sqrt(emit*beta)
    # height = math.sqrt(emit*gamma)
    # ang = 0.5*math.atan(2*alpha/(gamma-beta))

    # ellipse = Ellipse((0,0),width=width,height=height,angle=math.degrees(ang),linestyle=linstyle)

    # ax.add_patch(ellipse)

    # ellipse = Ellipse(xy=(157.18, 68.4705), width=0.036, height=0.012, 
    #                     edgecolor='r', fc='None', lw=2)
    # ax.add_patch(ellipse)
    N = 200
    x = np.zeros(N)
    xp = np.zeros(N)
    for i in range(N):
        theta = 20*i/(N*np.pi)
        x[i] = math.sqrt(emit*beta)*math.cos(theta)
        xp[i] = -math.sqrt(emit/beta)*(alpha*math.cos(theta)+math.sin(theta))

    # print(x)
    # print(xp)
    x = x*1e3
    xp = xp*1e3
    plt.plot(x,xp,linestyle=linstyle,color=color,label=label)

def save_2scat(data, name):
    # save the figures for the 3 histograms created by the x,y data given in 'data'

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    matplotlib.rc('font', **font)

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

    xn = shift_data(xn,xshift)
    yn = shift_data(yn,yshift)

    scatfigx = plt.figure()
    plt.plot(newbins,xn,'o',color='blue',label='horizontal')
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity')
    plt.xlim(xlimits)
    plt.title(name)
    plt.tight_layout()
    plt.savefig('histograms/'+name+'_x')

    scatfigy = plt.figure()
    plt.plot(newbins,yn,'o',color='green',label='vertical')
    plt.xlabel('y [mm]')
    plt.ylabel('Intensity')
    plt.xlim(xlimits)
    plt.title(name)
    plt.tight_layout()
    plt.savefig('histograms/'+name+'_y')

    # plt.savefig('histograms/z'+name)

    plt.close('all')

    return (data['#x'].std(),data['y'].std())


# various ploting and analysis tools for analysis of M4 beam line on 
# fermilab's Muon campus
# part of the SULI 2022 summer research project
# Author: Trevor Loe

from errno import ESPIPE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
from gekko import GEKKO
import os
from scipy.optimize import curve_fit
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
    # print(i)

    return df

def save_3hist(data, name):
    # save the figures for the 3 histograms created by the x,y, and z data given in 'data'

    binsno = 48
    bins = np.linspace(-24,24,binsno)
    xlimits = (-24,24)

    # x data
    figx = plt.figure()
    plt.hist(data['#x'], bins=bins,rwidth=0.85)
    # figure labels and stuff
    plt.xlabel('x [mm]')
    plt.ylabel('Events')
    plt.xlim(xlimits)
    plt.title(name)

    plt.savefig('histograms/x'+name)

    print('x std: ')
    print(data['#x'].std())

    # y data
    figy = plt.figure()
    plt.hist(data['y'], bins=bins,rwidth=0.85)
    # figure labels and stuff
    plt.xlabel('y [mm]')
    plt.ylabel('Events')
    plt.xlim(xlimits)
    plt.title(name)

    plt.savefig('histograms/y'+name)

    # z data
    figz = plt.figure()
    plt.hist(data['z'], bins=bins,rwidth=0.85)
    # figure labels and stuff
    plt.xlabel('z [mm]')
    plt.ylabel('Events')
    plt.xlim(xlimits)
    plt.title(name)

    plt.savefig('histograms/z'+name)

    plt.close('all')

    return (data['#x'].std(),data['y'].std())

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

def twistx(data):
    # compute the twist parameters for a given data set in form of a pandas dataframe
    # uses the essential math defining the twist parameters
    # needs position and momentum

    # computes twist parameters in x
    x = data['#x'].to_numpy()/1000
    px = data['Px'].to_numpy()/1000

    pz = data['Pz'].to_numpy()/1000
    #get mean velocity
    v = np.average(pz)

    # compute x prime
    xp = px/v

    Axx = np.std(x)**2
    Axpxp = np.std(xp)**2
    Axxp = np.average((x-np.average(x))*(xp-np.average(xp)))

    # compute emitance
    eps = math.sqrt(Axx*Axpxp - Axxp**2)
    beta = Axx/eps
    alpha = -Axxp/eps

    #print(np.std(x))
    #print(math.sqrt(beta*eps))


    return (alpha,beta,eps)

def twisty(data):
    # compute the twist parameters for a given data set in form of a pandas dataframe
    # uses the essential math defining the twist parameters
    # needs position and momentum

    # computes twist parameters in x
    y = data['y'].to_numpy()/1000
    py = data['Py'].to_numpy()/1000

    pz = data['Pz'].to_numpy()/1000
    #get mean velocity
    v = np.average(pz)

    # compute x prime
    yp = py/v

    Ayy = np.std(y)**2
    Aypyp = np.std(yp)**2
    Ayyp = np.average((y-np.average(y))*(yp-np.average(yp)))

    # compute emitance
    eps = math.sqrt(Ayy*Aypyp - Ayyp**2)
    beta = Ayy/eps
    alpha = -Ayyp/eps

    #print(np.std(x))
    #print(math.sqrt(beta*eps))


    return (alpha,beta,eps)


def thrsc(sig1,sig2,sig3,d1,d2,d3):
    # takes 3 sets of x data and uses the 3-screen method to compute beta_0, alpha_0, and eps
    # baseline equation for each dataset:
    # sig^2 = beta_0*eps - 2d*alpha_0*eps + d^2*eps*(1+alpha_0^2)/beta_0
    # data given as numpy arrays

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

    print(args)
    # alpha,beta,eps = opt.fsolve(sys,(1,10,2e-4),args=args)
    alpha,beta,eps = opt.fsolve(sys,(0.2,3,2e-6),args=args)

    return alpha,beta,eps


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
    alpha = m.Var(value=0.1)
    beta = m.Var(value=3)
    eps = m.Var(value=2e-6)

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
    print(bval*eval - 2*d1*aval*eval + d1**2*eval*(1+aval**2)/bval)
    print(sig1**2)

    return alpha.value,beta.value,eps.value

def disp_save_g4blhist(dist,dir,name):
    # creates a histogram from a distribution generated from g4beamline
    # displays the histogram and will display with plt.show() following
    # must have figure initialized 

    binsno = 48
    bins = np.linspace(-24,24,binsno)
    xlimits = (-24,24)

    # plt.figure()
    plt.hist(dist, bins=bins,rwidth=0.85)
    plt.xlabel('pos [mm]')
    plt.ylabel('Events')
    plt.xlim(xlimits)
    plt.title(name)

    loc = os.path.join(dir,name)
    plt.savefig(loc)

def hist_guassfit_g4bl(dist,name):
    # creates a histogram from the given distribution and fits a guassian to the bin heights 
    # must have plt figure initialized 

    binsno = 48
    bins = np.linspace(-24,24,binsno)
    xlimits = (-24,24)

    # plt.figure()
    n,bou,patches = plt.hist(dist, bins=bins,rwidth=0.85)
    # print(n)

    A,x0,sig,err = fit_guass(n)

    # print(A)
    # print(x0)
    # print(sig)
    # print(err)

    fitdata = guass(bins,A,x0,sig)
    plt.plot(bins,fitdata)

    plt.xlabel('pos [m]')
    plt.ylabel('Events')
    plt.title('Dist and fit of '+name)

    return dist.std(),sig


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
    # data = data - np.max(data)
    # data[data>0.1*np.min(data)]=0.1*np.min(data)
    # data = data - np.max(data)
    xdata = np.linspace(-24,24,len(data))
    # data,xdata = cutoff_noise(data,xdata)

    # print(data)
    
    parameters, covarience = curve_fit(guass,xdata,data,p0=(-1,-5,3))

    A = parameters[0]
    x0 = parameters[1]
    sig = parameters[2]

    # print(covarience)
    err = np.sqrt(np.diag(covarience))
    return (A,x0,sig,err)

def fit_plot_guass(data):
    # fits and plots a guassian to the numpy array 'data'
    # must have matplotlib figure initialized

    A,x0,sig,err = fit_guass(data)

    print(A)
    print(x0)
    print(sig)
    print(err)

    data = data - np.max(data)
    # data[data>0.1*np.min(data)]=0.1*np.min(data)
    # data = data - np.max(data)

    xdata = np.linspace(-24,24,len(data))
    fitdata = guass(xdata,A,x0,sig)

    plt.plot(xdata,data,'o',label='data')
    plt.plot(xdata,fitdata,'-',label='guassian fit')
    plt.legend()
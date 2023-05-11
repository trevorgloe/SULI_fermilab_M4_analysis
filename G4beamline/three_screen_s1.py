## file to test the 3 screen method on simulated data from G4beamline
# part of a SULI Summer 2022 project
# simulated data is for Fermilab's muon campus M4 line
# Author: Trevor Loe

import plot_fnc
import numpy as np

# first compute the twist parameters from the 'Control' detector data
control_data = plot_fnc.read_data('Control.txt')

(alpha,beta,eps) = plot_fnc.twistx(control_data)

print('in x:')
print('alpha = '+str(alpha))
print('beta = '+str(beta))
print('eps = '+str(eps))

# plot_fnc.save_3hist(control_data,'control')

print('in y:')
(alpha,beta,eps) = plot_fnc.twisty(control_data)

print('alpha = '+str(alpha))
print('beta = '+str(beta))
print('eps = '+str(eps))


# now pull the data from 924, 926 and 927
data924 = plot_fnc.read_data('MW924.txt')
data926 = plot_fnc.read_data('MW926.txt')
data927 = plot_fnc.read_data('MW927.txt')

print('in x:')
# convert to m
x924 = data924['#x'].to_numpy()
x926 = data926['#x'].to_numpy()
x927 = data927['#x'].to_numpy()

x924 = x924/1000
x926 = x926/1000
x927 = x927/1000
# print(x924)

# using same distances as were used in the simulated data (from G4beamline)
d = [1.52993,8.76993,14.86993]      # distance between Q924 and detectors MW924, MW926, and MW927
# d = [0.483,6.583,30.453]            # distance between Q926 and detectors MW926, MW927 and MW930
# d = [2.05755,25.92755,48.45755]       # distance between Q927 and detectors MW927, MW930, and MW932
d1 = d[0]
d2 = d[1]
d3 = d[2]


calpha,cbeta,ceps = plot_fnc.thrsc(x924,x926,x927,d1,d2,d3)
print('3 screen method:')
print('alpha = '+str(calpha))
print('beta = '+str(cbeta))
print('eps = '+str(ceps))

# do the same thing in y
print('in y:')

# convert to m
y924 = data924['y'].to_numpy()
y926 = data926['y'].to_numpy()
y927 = data927['y'].to_numpy()

y924 = y924/1000
y926 = y926/1000
y927 = y927/1000
# print(x924)

d1 = d[0]
d2 = d[1]
d3 = d[2]

calpha,cbeta,ceps = plot_fnc.thrsc(y924,y926,y927,d1,d2,d3)
print('3 screen method:')
print('alpha = '+str(calpha))
print('beta = '+str(cbeta))
print('eps = '+str(ceps))


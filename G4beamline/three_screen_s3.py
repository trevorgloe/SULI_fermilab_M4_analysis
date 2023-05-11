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

print('in y:')
(alpha,beta,eps) = plot_fnc.twisty(control_data)

print('alpha = '+str(alpha))
print('beta = '+str(beta))
print('eps = '+str(eps))


##### three screen method
print('3-screen method')

data927 = plot_fnc.read_data('MW927.txt')
data930 = plot_fnc.read_data('MW930.txt')
data932 = plot_fnc.read_data('MW932.txt')


#### do the 3-screen method in x
print('in x:')
x927 = data927['#x'].to_numpy()
x930 = data930['#x'].to_numpy()
x932 = data932['#x'].to_numpy()

x927 = x927/1000
x930 = x930/1000
x932 = x932/1000

#distances
# d = [1.52993,8.76993,14.86993]      # distance between Q924 and detectors MW924, MW926, and MW927
# d = [0.483,6.583,30.453]            # distance between Q926 and detectors MW926, MW927 and MW930
d = [2.05755,25.92755,48.45755]       # distance between Q927 and detectors MW927, MW930, and MW932
d1 = d[0]
d2 = d[1]
d3 = d[2]

calpha,cbeta,ceps = plot_fnc.thrsc(x927,x930,x932,d1,d2,d3)

print('alpha = '+str(calpha))
print('beta = '+str(cbeta))
print('eps = '+str(ceps))


#### do the 3-screen method in y
print('in y:')
y927 = data927['y'].to_numpy()
y930 = data930['y'].to_numpy()
y932 = data932['y'].to_numpy()

y927 = y927/1000
y930 = y930/1000
y932 = y932/1000

#distances
# d = [1.52993,8.76993,14.86993]      # distance between Q924 and detectors MW924, MW926, and MW927
# d = [0.483,6.583,30.453]            # distance between Q926 and detectors MW926, MW927 and MW930
d = [2.05755,25.92755,48.45755]       # distance between Q927 and detectors MW927, MW930, and MW932
d1 = d[0]
d2 = d[1]
d3 = d[2]

calpha,cbeta,ceps = plot_fnc.thrsc(y927,y930,y932,d1,d2,d3)

print('alpha = '+str(calpha))
print('beta = '+str(cbeta))
print('eps = '+str(ceps))
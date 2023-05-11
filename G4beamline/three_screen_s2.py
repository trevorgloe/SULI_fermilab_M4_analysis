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

print('3-screen method:')
#### 3-screen method in x
print('in x:')
data926 = plot_fnc.read_data('MW926.txt')
data927 = plot_fnc.read_data('MW927.txt')
data930 = plot_fnc.read_data('MW930.txt')

x926 = data926['#x'].to_numpy()
x927 = data927['#x'].to_numpy()
x930 = data930['#x'].to_numpy()

x926 = x926/1000
x927 = x927/1000
x930 = x930/1000

#distances
# d = [1.52993,8.76993,14.86993]      # distance between Q924 and detectors MW924, MW926, and MW927
d = [0.483,6.583,30.453]            # distance between Q926 and detectors MW926, MW927 and MW930
# d = [2.05755,25.92755,48.45755]       # distance between Q927 and detectors MW927, MW930, and MW932
d1 = d[0]
d2 = d[1]
d3 = d[2]

calpha,cbeta,ceps = plot_fnc.thrsc(x926.std(),x927.std(),x930.std(),d1,d2,d3)

print('alpha = '+str(calpha))
print('beta = '+str(cbeta))
print('emit = '+str(ceps))


#### 3-screen method in y
print('in y:')

y926 = data926['y'].to_numpy()
y927 = data927['y'].to_numpy()
y930 = data930['y'].to_numpy()

y926 = y926/1000
y927 = y927/1000
y930 = y930/1000

#distances
d1 = d[0]
d2 = d[1]
d3 = d[2]

calpha,cbeta,ceps = plot_fnc.thrsc(y926.std(),y927.std(),y930.std(),d1,d2,d3)

print('alpha = '+str(calpha))
print('beta = '+str(cbeta))
print('emit = '+str(ceps))


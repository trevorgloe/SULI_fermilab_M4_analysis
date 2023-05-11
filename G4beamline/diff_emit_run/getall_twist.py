### file to calculate twist parameters from G4beamline distributions with new emittance


import plot_fnc
import numpy as np
import pandas as pd


##### Downstream of Q924
data924 = plot_fnc.read_data('Control_924.txt')

(alpha1x,beta1x,eps1x) = plot_fnc.twistx(data924)
(alpha1y,beta1y,eps1y) = plot_fnc.twisty(data924)

print('Dist downstream of Q924')
print('in x:')
print('alpha = '+str(alpha1x))
print('beta = '+str(beta1x))
print('eps = '+str(eps1x))
print('in y:')
print('alpha = '+str(alpha1y))
print('beta = '+str(beta1y))
print('eps = '+str(eps1y))


##### Upstream of Q925
data925 = plot_fnc.read_data('Control_925.txt')

(alpha2x,beta2x,eps2x) = plot_fnc.twistx(data925)
(alpha2y,beta2y,eps2y) = plot_fnc.twisty(data925)

print('Dist upstream of Q925')
print('in x:')
print('alpha = '+str(alpha2x))
print('beta = '+str(beta2x))
print('eps = '+str(eps2x))
print('in y:')
print('alpha = '+str(alpha2y))
print('beta = '+str(beta2y))
print('eps = '+str(eps2y))


##### Downstream of Q926
data926 = plot_fnc.read_data('Control_926.txt')

(alpha3x,beta3x,eps3x) = plot_fnc.twistx(data926)
(alpha3y,beta3y,eps3y) = plot_fnc.twisty(data926)

print('Dist downstream of Q926')
print('in x:')
print('alpha = '+str(alpha3x))
print('beta = '+str(beta3x))
print('eps = '+str(eps3x))
print('in y:')
print('alpha = '+str(alpha3y))
print('beta = '+str(beta3y))
print('eps = '+str(eps3y))


##### Downstream of Q927
data927 = plot_fnc.read_data('Control_927.txt')

(alpha4x,beta4x,eps4x) = plot_fnc.twistx(data927)
(alpha4y,beta4y,eps4y) = plot_fnc.twisty(data927)

print('Dist downstream of Q927')
print('in x:')
print('alpha = '+str(alpha4x))
print('beta = '+str(beta4x))
print('eps = '+str(eps4x))
print('in y:')
print('alpha = '+str(alpha4y))
print('beta = '+str(beta4y))
print('eps = '+str(eps4y))


##### Upstream of Q930
data930 = plot_fnc.read_data('Control_930.txt')

(alpha5x,beta5x,eps5x) = plot_fnc.twistx(data930)
(alpha5y,beta5y,eps5y) = plot_fnc.twisty(data930)

print('Dist upstream of Q930')
print('in x:')
print('alpha = '+str(alpha5x))
print('beta = '+str(beta5x))
print('eps = '+str(eps5x))
print('in y:')
print('alpha = '+str(alpha5y))
print('beta = '+str(beta5y))
print('eps = '+str(eps5y))
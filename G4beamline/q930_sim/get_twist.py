## file to get twist parameters from control distribution at Q930

import plot_fnc

data = plot_fnc.read_data('Control.txt')
(alphax,betax,epsx) = plot_fnc.twistx(data)

(alphay,betay,epsy) = plot_fnc.twisty(data)

print('Control distribution:')
print('in x:')
(alphax,betax,epsx) = plot_fnc.twistx(data)

print('alpha = '+str(alphax))
print('beta = '+str(betax))
print('eps = '+str(epsx))


print('in y:')
(alphay,betay,epsy) = plot_fnc.twisty(data)

print('alpha = '+str(alphay))
print('beta = '+str(betay))
print('eps = '+str(epsy))
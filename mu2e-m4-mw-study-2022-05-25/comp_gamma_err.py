## just compute gamma and uncertainty

from math import *

# Q924
print('Q924')
ax = 0.026369035
bx = 8.207675856
gx = (1 + ax**2)/bx
dax = 0.050995962
dbx = 0.195197968
dnum = 2*ax*dax
dgx = gx*sqrt((dnum/(1+ax**2))**2 + (dbx/bx)**2)

print('for x')
print(str(gx)+' +/- '+str(dgx))

ay = 0.311146701
by = 21.85963074
gy = (1 + ay**2)/by
day = 0.023209183
dby = 0.710382838
dnum = 2*ay*day
dgy = gy*sqrt((dnum/(1+ay**2))**2 + (dby/by)**2)

print('for y')
print(str(gy)+' +/- '+str(dgy))

# Q926
print('Q926')
ax = 0.717882596
bx = 18.31005578
gx = (1 + ax**2)/bx
dax = 0.047533715
dbx = 0.304438726
dnum = 2*ax*dax
dgx = gx*sqrt((dnum/(1+ax**2))**2 + (dbx/bx)**2)

print('for x')
print(str(gx)+' +/- '+str(dgx))

ay = -1.046614254
by = 17.83391912
gy = (1 + ay**2)/by
day = 0.29728509
dby = 3.381057768
dnum = 2*ay*day
dgy = gy*sqrt((dnum/(1+ay**2))**2 + (dby/by)**2)

print('for y')
print(str(gy)+' +/- '+str(dgy))

# Q927
print('Q927')
ax = -0.216875555
bx = 17.7453019
gx = (1 + ax**2)/bx
dax = 0.076598443
dbx = 0.770030522
dnum = 2*ax*dax
dgx = gx*sqrt((dnum/(1+ax**2))**2 + (dbx/bx)**2)

print('for x')
print(str(gx)+' +/- '+str(dgx))

ay = -0.535437087
by = 30.89687068
gy = (1 + ay**2)/by
day = 0.119936427
dby = 2.953575483
dnum = 2*ay*day
dgy = gy*sqrt((dnum/(1+ay**2))**2 + (dby/by)**2)

print('for y')
print(str(gy)+' +/- '+str(dgy))
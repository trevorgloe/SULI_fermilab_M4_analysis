## calculate transport matrices and propogate twist parameters 

import numpy as np
from math import *

def twist_mat(m):
    # make twist param tranfer matrix from normal 2x2 matrix
    m11 = m[0,0]
    m12 = m[0,1]
    m21 = m[1,0]
    m22 = m[1,1]

    twist_mat = np.array([[m11**2, -2*m11*m12, m12**2],
                        [-m11*m21, 1+2*m12*m21, -m12*m22],
                        [m21**2, -2*m21*m22, m22**2]])

    return twist_mat

################################### propogate between Q924 and Q926
d1 = 4.32622    # distance betweem Q924 and Q925
l925 = 0.45720  # length of Q925 quad
g925 = 2.230560618882525    # gradient for Q925 quad
k925 = g925 * 0.2998/ 8.89  # kappa for Q925 quad (m^-2)

d2 = 2.46154    # distance between Q925 and Q926
l926 = 0.45720
g926 = 6.666434753791199    # gradient for Q926 quad
k926 = g926 * 0.2998/ 8.89  # kappa for Q926 quad

# print(k925)
# print(k926)

# first drift matrix
m1 = np.array([[1,d1],[0,1]])
# print(m1)

# x and y matrices for q925
m2x = np.array([[cosh(sqrt(k925)*l925),1/sqrt(k925)*sinh(sqrt(k925)*l925)],
                [sqrt(k925)*sinh(sqrt(k925)*l925),cosh(sqrt(k925)*l925)]])

m2y = np.array([[cos(sqrt(k925)*l925),1/sqrt(k925)*sin(sqrt(k925)*l925)],
                [-sqrt(k925)*sin(sqrt(k925)*l925),cos(sqrt(k925)*l925)]])
# using thin lens
# m2x = np.array([[1,0],[k925*l925,1]])
# m2y = np.array([[1,0],[-k925*l925,1]])

# print(m2x)
# drift from q925 to q926
m3 = np.array([[1,d2],[0,1]])
# print(m3)

# x and y matrices for q926
m4x = np.array([[cos(sqrt(k926)*l926),1/sqrt(k926)*sin(sqrt(k926)*l926)],
                [-sqrt(k926)*sin(sqrt(k926)*l926),cos(sqrt(k926)*l926)]])

m4y = np.array([[cosh(sqrt(k926)*l926),1/sqrt(k926)*sinh(sqrt(k926)*l926)],
                [sqrt(k926)*sinh(sqrt(k926)*l926),cosh(sqrt(k926)*l926)]])
# m4x = np.array([[1,0],[-k926*l926,1]])
# m4y = np.array([[1,0],[k926*l926,1]])

# print(m4x)
# get total transport matrix
mxtot924 = m4x@m3@m2x@m1
mytot924 = m4y@m3@m2y@m1

twist_matx = twist_mat(mxtot924)
twist_maty = twist_mat(mytot924)

# initial twist parameters
ax = 0.026369035
bx = 8.207675856
gx = (1 + ax**2)/bx
abgx = np.array([bx,ax,gx])
ay = 0.311146701
by = 21.85963074
gy = (1 + ay**2)/by
abgy = np.array([by,ay,gy])

print('Propogating from Q924 to Q926')
print('x: ')
print(abgx)
print(twist_matx)
print('multiplied:')
print(twist_matx@abgx)
print('y: ')
print(abgy)
print(twist_maty)
print('multiplied:')
print(twist_maty@abgy)


################################### propogate between Q926 and Q927
d1 = 4.02491    # distance betweem Q924 and Q925
l927 = 0.45720  # length of Q927 quad
g927 = 2.0139631110340424    # gradient for Q927 quad (T/m)
k927 = g927 * 0.2998/ 8.89  # kappa for Q927 quad (m^-2)

# first drift matrix
m1 = np.array([[1,d1],[0,1]])

# x and y matrices for q925
m2x = np.array([[cosh(sqrt(k927)*l927),1/sqrt(k927)*sinh(sqrt(k927)*l927)],
                [sqrt(k927)*sinh(sqrt(k927)*l927),cosh(sqrt(k927)*l927)]])

m2y = np.array([[cos(sqrt(k927)*l927),1/sqrt(k927)*sin(sqrt(k927)*l927)],
                [-sqrt(k927)*sin(sqrt(k927)*l927),cos(sqrt(k927)*l927)]])

mxtot926 = m2x@m1
mytot926 = m2y@m1

twist_matx = twist_mat(mxtot926)
twist_maty = twist_mat(mytot926)

# initial twist parameters
ax = 0.717882596
bx = 18.31005578
gx = (1 + ax**2)/bx
abgx = np.array([bx,ax,gx])
ay = -1.046614254
by = 17.83391912
gy = (1 + ay**2)/by
abgy = np.array([by,ay,gy])

print('Propogating from Q926 to Q927')
print('x: ')
print(abgx)
print(twist_matx)
print('multiplied:')
print(twist_matx@abgx)
print('y: ')
print(abgy)
print(twist_maty)
print('multiplied:')
print(twist_maty@abgy)


################################### propogate between Q924 and Q927

mxtot = mxtot926@mxtot924
mytot = mytot926@mytot924

mx24to27 = mxtot
my24to27 = mytot

twist_matx = twist_mat(mxtot)
twist_maty = twist_mat(mytot)

# initial twist parameters
ax = 0.026369035
bx = 8.207675856
gx = (1 + ax**2)/bx
abgx = np.array([bx,ax,gx])
ay = 0.311146701
by = 21.85963074
gy = (1 + ay**2)/by
abgy = np.array([by,ay,gy])

print('Propogating from Q924 to Q927')
print('x: ')
print(abgx)
print(twist_matx)
print('multiplied:')
print(twist_matx@abgx)
print('y: ')
print(abgy)
print(twist_maty)
print('multiplied:')
print(twist_maty@abgy)


################################### propogate between Q924 and Q925
d1 = 4.32622    # distance betweem Q924 and Q925

# first drift matrix
m1 = np.array([[1,d1],[0,1]])

mxtot925 = m1
mytot925 = m1

twist_matx = twist_mat(mxtot925)
twist_maty = twist_mat(mytot925)

# initial twist parameters
ax = 0.026369035
bx = 8.207675856
gx = (1 + ax**2)/bx
abgx = np.array([bx,ax,gx])
ay = 0.311146701
by = 21.85963074
gy = (1 + ay**2)/by
abgy = np.array([by,ay,gy])

print('Propogating from Q924 to Q925')
print('x: ')
print(abgx)
print(twist_matx)
print('multiplied:')
print(twist_matx@abgx)
print('y: ')
print(abgy)
print(twist_maty)
print('multiplied:')
print(twist_maty@abgy)


################################### propogate between Q924 and Q930
d1 = 8.29794    # distance betweem Q927 and Q928
d2 = 5.50139    # distance between Q928 and Q929
d3 = 8.00374    # distance between Q929 and Q930
g928 = 6.153393792293593    # gradient for Q928 quad (T/m)
k928 = g928 * 0.2998/ 8.89  # kappa for Q928 quad (m^-2)
g929 = 5.580790039177821    # gradient for Q929 quad (T/m)
k929 = g929 * 0.2998/ 8.89  # kappa for Q929 quad (m^-2)
l928 = 640.08/1e3
l929 = 457.20/1e3


# first drift matrix
md1 = np.array([[1,d1],[0,1]])

# matrix for Q928 - focusing in y, defocusing in x
m928x = np.array([[cosh(sqrt(k928)*l928),1/sqrt(k928)*sinh(sqrt(k928)*l928)],
                [sqrt(k928)*sinh(sqrt(k928)*l928),cosh(sqrt(k928)*l928)]])

m928y = np.array([[cos(sqrt(k928)*l928),1/sqrt(k928)*sin(sqrt(k928)*l928)],
                [-sqrt(k928)*sin(sqrt(k928)*l928),cos(sqrt(k928)*l928)]])

# second drift matrix
md2 = np.array([[1,d2],[0,1]])

# matrix for Q929 - focusing in x, defocusing in y
m929x = np.array([[cos(sqrt(k929)*l929),1/sqrt(k929)*sin(sqrt(k929)*l929)],
                [-sqrt(k929)*sin(sqrt(k929)*l929),cos(sqrt(k929)*l929)]])

m929y = np.array([[cosh(sqrt(k929)*l929),1/sqrt(k929)*sinh(sqrt(k929)*l929)],
                [sqrt(k929)*sinh(sqrt(k929)*l929),cosh(sqrt(k929)*l929)]])

# third drift matrix
md3 = np.array([[1,d3],[0,1]])

mxtot930 = md3@m929x@md2@m928x@md1@mx24to27
mytot930 = md3@m929y@md2@m928y@md1@my24to27

twist_matx = twist_mat(mxtot930)
twist_maty = twist_mat(mytot930)

# initial twist parameters
ax = 0.026369035
bx = 8.207675856
gx = (1 + ax**2)/bx
abgx = np.array([bx,ax,gx])
ay = 0.311146701
by = 21.85963074
gy = (1 + ay**2)/by
abgy = np.array([by,ay,gy])

print('Propogating from Q924 to Q930')
print('x: ')
print(abgx)
print(twist_matx)
print('multiplied:')
print(twist_matx@abgx)
print('y: ')
print(abgy)
print(twist_maty)
print('multiplied:')
print(twist_maty@abgy)
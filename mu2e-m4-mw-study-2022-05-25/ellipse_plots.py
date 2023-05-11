## make some phase-sapce ellipses

import plot_fnc

# import modules
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

dir = os.getcwd()
figpath = os.path.join(dir,'ellipses_comp')
  
# adjust figure and assign coordinates
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# alx = 0.717882596
# bex = 18.31005578
# emitx = 3.55e-7
# # pp1 = plt.Rectangle((0.2, 0.75),
# #                     0.4, 0.15)
  
# # # pp2 = plt.Circle((0.7, 0.2), 0.15)
  
# # pp3 = Ellipse(xy=(0, 0), width=0.36, height=0.12, angle=10, 
# #                         edgecolor='r', fc='None', lw=2)
  
# # # depict illustrations
# # ax.add_patch(pp1)
# # # ax.add_patch(pp2)
# # ax.add_patch(pp3)
# # plt.axis('scaled')
# plot_fnc.make_ellipse(alx,bex,emitx,':','red',ax)
# plt.axis('equal')

########## Q924 x
# from data
alpha = 0.026369035
beta = 8.207675856
emit = 3.61E-07

fig1 = plt.figure()
plot_fnc.make_ellipse(alpha,beta,emit,'-','blue','3-screen method')

# from G4beamline
alpha_sim = 0.197171097349287
beta_sim = 4.01386017610141
emit_sim = 2.43662163925914E-07
plot_fnc.make_ellipse(alpha_sim,beta_sim,emit_sim,':','red','G4beamline')
plt.axis('equal')
plt.xlabel('x [mm]')
plt.ylabel("x' [mrad]")
plt.legend()

plt.savefig(os.path.join(figpath,'Q924x'))

########## Q924 y
# from data
alpha = 0.311146701
beta = 21.85963074
emit = 4.20E-07

fig1 = plt.figure()
plot_fnc.make_ellipse(alpha,beta,emit,'-','blue','3-screen method')

# from G4beamline
alpha_sim = 0.258854264230636
beta_sim = 20.6342049471516
emit_sim = 2.50246217970826E-07
plot_fnc.make_ellipse(alpha_sim,beta_sim,emit_sim,':','red','G4beamline')
plt.axis('equal')
plt.xlabel('y [mm]')
plt.ylabel("y' [mrad]")
plt.legend()

plt.savefig(os.path.join(figpath,'Q924y'))

########## Q926 x
# from data
alpha = 0.717882596
beta = 18.31005578
emit = 3.55E-07

fig1 = plt.figure()
plot_fnc.make_ellipse(alpha,beta,emit,'-','blue','3-screen method')

# from G4beamline
alpha_sim = -0.408217898
beta_sim = 20.21570002
emit_sim = 0.000000244
plot_fnc.make_ellipse(alpha_sim,beta_sim,emit_sim,':','red','G4beamline')
plt.axis('equal')
plt.xlabel('x [mm]')
plt.ylabel("x' [mrad]")
plt.legend()

plt.savefig(os.path.join(figpath,'Q926x'))

########## Q926 y
# from data
alpha = -1.046614254
beta = 17.83391912
emit = 4.42E-07

fig1 = plt.figure()
plot_fnc.make_ellipse(alpha,beta,emit,'-','blue','3-screen method')

# from G4beamline
alpha_sim = -1.267828357
beta_sim = 16.76214356
emit_sim = 0.00000025
plot_fnc.make_ellipse(alpha_sim,beta_sim,emit_sim,':','red','G4beamline')
plt.axis('equal')
plt.xlabel('y [mm]')
plt.ylabel("y' [mrad]")
plt.legend()

plt.savefig(os.path.join(figpath,'Q926y'))

########## Q927 x
# from data
alpha = -0.216875555
beta = 17.7453019
emit = 3.54E-07

fig1 = plt.figure()
plot_fnc.make_ellipse(alpha,beta,emit,'-','blue','3-screen method')

# from G4beamline
alpha_sim = -1.45340134854329
beta_sim = 25.5134046926539
emit_sim = 0.000000244
plot_fnc.make_ellipse(alpha_sim,beta_sim,emit_sim,':','red','G4beamline')
plt.axis('equal')
plt.xlabel('x [mm]')
plt.ylabel("x' [mrad]")
plt.legend()

plt.savefig(os.path.join(figpath,'Q927x'))

########## Q927 y
# from data
alpha = -0.535437087
beta = 30.89687068
emit = 4.35E-07

fig1 = plt.figure()
plot_fnc.make_ellipse(alpha,beta,emit,'-','blue','3-screen method')

# from G4beamline
alpha_sim = -1.01070787133377
beta_sim = 30.9083758291278
emit_sim = 0.00000025
plot_fnc.make_ellipse(alpha_sim,beta_sim,emit_sim,':','red','G4beamline')
plt.axis('equal')
plt.xlabel('y [mm]')
plt.ylabel("y' [mrad]")
plt.legend()

plt.savefig(os.path.join(figpath,'Q927y'))

########## Q925 y
# from data
alpha = 0.2177376016700482
beta = 37.31592715244996
emit = 0.39618309713741744e-06

fig1 = plt.figure()
plot_fnc.make_ellipse(alpha,beta,emit,'-','blue','quad-scan method')

# from G4beamline
alpha_sim = 0.00981988243864445 
beta_sim = 19.3391390512299 
emit_sim = 0.250247e-06 
plot_fnc.make_ellipse(alpha_sim,beta_sim,emit_sim,':','red','G4beamline')
plt.axis('equal')
plt.xlabel('y [mm]')
plt.ylabel("y' [mrad]")
plt.legend()

plt.savefig(os.path.join(figpath,'Q925y'))

########## Q930 x
# from data
alpha = 0.059558935736502075
beta = 115.07253009690645
emit = 0.3386447438645042e-06

fig1 = plt.figure()
plot_fnc.make_ellipse(alpha,beta,emit,'-','blue','quad-scan method')

# from G4beamline
alpha_sim = -0.10214658060421
beta_sim = 255.47986172204 
emit_sim = 0.24366e-06
plot_fnc.make_ellipse(alpha_sim,beta_sim,emit_sim,':','red','G4beamline')
plt.axis('equal')
plt.xlabel('x [mm]')
plt.ylabel("x' [mrad]")
plt.legend()

plt.savefig(os.path.join(figpath,'Q930x'))

plt.show()
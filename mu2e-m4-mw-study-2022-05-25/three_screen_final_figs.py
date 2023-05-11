## file to create general figure for all twist parameters extracted from 3-screen method

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# create location to save it
dir = os.getcwd()
figpath = os.path.join(dir,'final_three_screen')

data = pd.read_csv('all_three_screen_results.csv',index_col=0)

# print(data)
dist = np.array([105010.07e-3,113297.54e-3,117822.45e-3])

# q924 = data.loc['Q924',:].to_numpy()
# q926 = data.loc['Q926',:].to_numpy()
# q927 = data.loc['Q927',:].to_numpy()

# all x twist parameters
all_alphax = data.loc[:,'alpha_x']
all_alphaxerr = data.loc[:,'dalpha_x']
all_betax = data.loc[:,'beta_x']
all_betaxerr = data.loc[:,'dbeta_x']
all_emitx = data.loc[:,'emit_x']
all_emitxerr = data.loc[:,'demit_x']

# all y twist parameters
all_alphay = data.loc[:,'alpha_y']
all_alphayerr = data.loc[:,'dalpha_y']
all_betay = data.loc[:,'beta_y']
all_betayerr = data.loc[:,'dbeta_y']
all_emity = data.loc[:,'emit_y']
all_emityerr = data.loc[:,'demit_y']


fig1 = plt.figure()
plt.errorbar(dist,all_alphax,all_alphaxerr,marker='o',linestyle='',color='blue',capsize=4,label='Horizontal plane')
plt.errorbar(dist,all_alphay,all_alphayerr,marker='o',linestyle='',color='red',capsize=4,label='Vertical plane')
plt.xlabel('Distance [m]')
plt.ylabel('$\\alpha$')
plt.xlim([100,130])
plt.legend()
plt.savefig(os.path.join(figpath,'all_alpha'))

fig2 = plt.figure()
plt.errorbar(dist,all_betax,all_betaxerr,marker='o',linestyle='',color='blue',capsize=4,label='Horizontal plane')
plt.errorbar(dist,all_betay,all_betayerr,marker='o',linestyle='',color='red',capsize=4,label='Vertical plane')
plt.xlabel('Distance [m]')
plt.ylabel('$\\beta$ [m]')
plt.xlim([100,130])
plt.legend()
plt.savefig(os.path.join(figpath,'all_beta'))

fig3 = plt.figure()
plt.errorbar(dist,all_emitx*1e6,all_emitxerr*1e6,marker='o',linestyle='',color='blue',capsize=4,label='Horizontal plane')
plt.errorbar(dist,all_emity*1e6,all_emityerr*1e6,marker='o',linestyle='',color='red',capsize=4,label='Vertical plane')
plt.xlabel('Distance [m]')
plt.ylabel('Emitance [$\mu$m]')
plt.xlim([100,130])
plt.ylim([0,0.5])
plt.legend()
plt.savefig(os.path.join(figpath,'all_emit'))

plt.show()

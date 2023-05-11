## script to change the current of a selected magnet in the G4bl file
# can be executed from the command line with an argument given that will be the current which
# magnet Q925 is changed to

import fileinput

# f = open('G4_M4_Mu2e_03_c.g4bl',"r+")

# for line in f.readlines():
#     print(line)
#     if line.find('param GQ925')==-1:
#         continue
#     else:
#         f.write('param GQ925=0')

import sys

newI = sys.argv[1] 

# newI = input('New current:')

cnt = 0
for line in fileinput.input('G4_M4_Mu2e_03.g4bl', inplace=True):
    if line.find('param GQ925=')==-1:
        print(line,end='')
    else:
        if line.find('#')==0:
            print(line,end='')
        else:
            cnt = cnt+1
            print('  param GQ925='+str(newI))

fileinput.close()
# print(cnt)
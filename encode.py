import os
from decimal import Decimal
import re

userdirs = os.listdir('./')
loca2code={}
code2loca={}
code = 0
for userdir in userdirs:
    if not os.path.isdir(userdir):
        continue
    filenames = os.listdir(userdir + '/Trajectory/')
    matchlist = []
    for i in range(len(filenames)):
        if re.match('^.*txt$',filenames[i]):
            matchlist.append(i)
    filenames = [filenames[index] for index in matchlist]
    filepaths = [userdir + '/Trajectory/' + filename for filename in filenames]
    for txt in filepaths:
        f = open(txt,'r').read().split('\n')
        for loca in f:
            if len(loca)!=0:
                if loca not in loca2code:
                    loca2code[loca] = code
                    code+=1
    print(userdir+'done\n')
floca2code = open('loca2code.txt','w')
for loca in loca2code:
    floca2code.write(loca + " " + str(loca2code[loca]) + '\n')
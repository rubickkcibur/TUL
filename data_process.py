import os
from decimal import Decimal
import re

def readfile(path):
    lines = open(path,'r',encoding='utf-8').read().split('\n')
    T = []
    for i in range(6,len(lines)):
        if len(lines[i]) == 0:
            break
        line = lines[i].split(',')
        data = [Decimal(float(d)).quantize(Decimal("0.000")) for d in line[:2]]
        T.append(data[:2])
    return T

userdirs = os.listdir('./')
for userdir in userdirs:
    if not os.path.isdir(userdir):
        continue
    filenames = os.listdir(userdir + '/Trajectory/')
    pltlist = []
    for i in range(len(filenames)):
        if re.match('^.*plt$',filenames[i]):
            pltlist.append(i)
    filenames = [filenames[index] for index in pltlist]
    filepaths = [userdir + '/Trajectory/' + filename for filename in filenames]
    for plt in filepaths:
        T = readfile(plt)
        f = open(plt+'.txt','w')
        for tuple in T:
            f.write(str(tuple[0])+','+str(tuple[1])+'\n')
    print(userdir + "done")
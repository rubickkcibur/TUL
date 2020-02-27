import os
from decimal import Decimal
import re

pathlen=200

loca2code = {}
fmap = open('loca2code.txt','r').read().split('\n')
for line in fmap:
    if len(line) != 0:
        tuple = line.split()
        loca2code[tuple[0]] = tuple[1]

userdirs = os.listdir('./')
for userdir in userdirs:
    cnt = 0
    if not os.path.isdir(userdir):
        continue
    if userdir[0] == '.':
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
        outdir = 'out/' + userdir + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fout = open(outdir + str(cnt) + '.txt', 'w')
        cnt += 1
        for loca in f:
            if len(loca)!=0:
                fout.write(loca2code[loca] + ' ')
    print(userdir + 'done\n')
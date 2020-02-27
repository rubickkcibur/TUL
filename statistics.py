import os

dirlist = os.listdir('out')

total = 0

for dir in dirlist:
    filelist = os.listdir('out/'+dir)
    total+=len(filelist)

print(total)
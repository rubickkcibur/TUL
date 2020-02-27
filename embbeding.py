import gensim
import os

class MySetences():
    def __init__(self,filepath):
        self.userdirs = ['out/' + dir for dir in os.listdir(filepath)]
    def __iter__(self):
        for userdir in self.userdirs:
            filenames = [userdir + '/' + filename for filename in os.listdir(userdir)]
            for file in filenames:
                yield open(file,'r').read().split()

model = gensim.models.Word2Vec(MySetences('out/'),min_count=1,size=256,window=16)

model.save('embbeding.model')


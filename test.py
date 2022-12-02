import numpy as np
l = np.array([np.array([0,0,1,0]), np.array([0,0,1,0]), np.array([1,0,0,0])])

def vote(l):
    new_l = np.zeros((1, l.shape[1]))
    new_l[0,np.argmax(np.sum(l, axis=0))] = 1
    return new_l

print(vote(l))
import numpy as np
import torch
import scipy
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity

directory = '/Users/hayoungsong/Documents/_postdoc/modelbrain/github'
filename = 'sub-001_sitcomep1'
sys.path.append(directory+'/model')
from mindy import mindy

def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))

#######################################################
# load fMRI & input time series
#######################################################
ts = scipy.io.loadmat(directory+'/data/song2023elife/fmri/sub-001/sitcomep1.mat')['ts'].T
ts = scipy.stats.zscore(ts, 1)
input = np.load(directory+'/data/song2023elife/input/sitcomep1_input_pc.npy').T
# they should be normalized across time per parcel or per input feature

nParcel = ts.shape[0]
nInput  = input.shape[0]
nT      = ts.shape[1]

print('ts:    '+str(ts.shape))
print('input: '+str(input.shape))

#######################################################
# load trained model parameters
#######################################################
model = torch.load(directory+'/output/'+filename+'.pth', weights_only=False)

#######################################################
# compare parameters with descriptive statistics
#######################################################
# model W & functional connectivity
fc_z = conv_r2z(np.corrcoef(ts))
np.fill_diagonal(fc_z, 0)

ww = model.to_numpy(model.pW*model.W)
w_und = np.zeros(ww.shape)+np.nan
for i1 in range(nParcel-1):
    for i2 in range(i1+1, nParcel):
        w_und[i1,i2] = (ww[i1,i2] + ww[i2,i1])/2
w_cos = cosine_similarity(w_und[np.where(~np.isnan(w_und))].reshape(1,-1), fc_z[np.where(~np.isnan(w_und))].reshape(1,-1))[0][0]
print('model W & functional connectivity: cosine similarity = '+str(np.round(w_cos,4)))


# model B & encoding coefficient
x_aug = np.vstack([input, np.ones((1, input.shape[1]))])
w_aug = ts @ np.linalg.pinv(x_aug)
enc = w_aug[:, :-1]

b_cos = cosine_similarity(enc.flatten().reshape(1,-1), model.to_numpy(model.pB*model.beta).flatten().reshape(1,-1))[0][0]
print('model B & encoding coefficient: cosine similarity = '+str(np.round(b_cos,4)))



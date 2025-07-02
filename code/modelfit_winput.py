import numpy as np
import torch
import scipy
import sys
import os

directory = '/set/directory' # ****** set directory *****
subject   = 'sub-001'
condition = 'sitcomep1'
filename = condition + '_' + subject
sys.path.append(directory+'/model')
from mindy import mindy

#######################################################
# setting
#######################################################
niter = 2500      # training iteration
batch_size = 300
nFut = 5000       # number of forward simulation

seed = 1
torch.manual_seed(seed), np.random.seed(seed)

if os.path.exists(directory+'/output')==False:
    os.mkdir(directory+'/output')

#######################################################
# load fMRI & input time series
#######################################################
ts = scipy.io.loadmat(directory+'/data/song2023elife/fmri/'+subject+'/'+condition+'.mat')['ts'].T
ts = scipy.stats.zscore(ts, 1)
input = np.load(directory+'/data/song2023elife/input/'+condition+'_input_pc.npy').T
# they should be normalized across time per parcel or per input feature

nParcel = ts.shape[0]
nInput  = input.shape[0]
nT      = ts.shape[1]

print('ts:    '+str(ts.shape))
print('input: '+str(input.shape))

#######################################################
# run model
#######################################################
model = mindy(nParcel, nInput)
# if running the model without input: model = mindy(nParcel, 0)

for iter in range(niter):
    t_ = np.random.permutation(nT - batch_size)[0]
    y, yhat, J = model.forward(ts[:, t_:t_ + batch_size], input[:, t_:t_ + batch_size])
    # if running the model without input: y, yhat, J = model.forward(ts[:, t_:t_ + batch_size], [])
    model.update_weights(J)
Xpred, pW, pD, pB = model.predict(ts, input)
torch.save(model, directory+'/output/'+filename+'.pth')

# prediction of next time step
y = ts[:, 1:]
y_pred = model.to_numpy(Xpred)
ss_total = np.sum((y - np.mean(y)) ** 2)  # Total variance
ss_residual = np.sum((y - y_pred) ** 2)   # Residual variance
r_squared = 1 - (ss_residual / ss_total)
print('rsq: ' + str(r_squared)) # rsquare would be near zero with the random data, but with an actual fMRI data, rsquare reaches near 0.5

# trained model parameters
alpha = model.to_numpy(model.alpha)
b = model.b
W = model.to_numpy(model.W)
Decay = model.to_numpy(model.Decay)
beta = model.to_numpy(model.beta)

#######################################################
# forward simulation: find attractors
#######################################################
xpred_time_start = np.zeros((nParcel, nT))
xpred_time_end = np.zeros((10, nParcel, nT))
term_b_ = np.zeros((nParcel, nT))
term_w_ = np.zeros((nParcel, nT))
term_d_ = np.zeros((nParcel, nT))
for t_ in range(nT):
    if np.mod(t_, 100)==0:
        print('    '+str(t_)+' / '+str(nT))
    # with brain activity at each time step as an initial point, forward simulate nFut times
    for t in range(nFut):
        if t ==0:
            x_base = ts[:,t_]
            m_base = input[:,t_]
        else:
            x_base = x_pred
        psi_x = np.sqrt(alpha**2 + (b * x_base + 0.5)**2) - np.sqrt(alpha**2 + (b * x_base - 0.5)**2)
        term_W = pW * W @ psi_x
        term_D = (-1) * (pD * Decay * x_base)
        x_pred = x_base + term_W + term_D
        if t==0:
            term_B = pB * beta @ m_base
            x_pred = x_base + term_W + term_D + term_B
            term_b_[:,t_] = term_B
            term_w_[:,t_] = term_W
            term_d_[:,t_] = term_D
            xpred_time_start[:,t_] = x_pred
        if t>=nFut-10:
            xpred_time_end[t-(nFut-10),:,t_] = x_pred

fixed_points = np.zeros((nParcel, nT)) + np.nan
for t_ in range(nT):
    if np.all(np.abs(np.diff(xpred_time_end, axis=0))<1e-6):
        fixed_points[:, t_] = xpred_time_end[-1,:,t_]
print(str(len(np.where(np.isnan(fixed_points[0,:]))[0]))+ ' out of '+str(nT)+' not converged to a fixed point')

np.savez_compressed(directory+'/output/'+filename,
                    pW=pW, pD=pD, pB=pB, rsq=r_squared, xpred_time_start=xpred_time_start, xpred_time_end=xpred_time_end,
                    fixed_points=fixed_points, term_W = term_w_, term_D = term_d_, term_B = term_b_)

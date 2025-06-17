import numpy as np
import torch
import scipy
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import os
sys.path.append('/Users/hayoungsong/Documents/_postdoc/modelbrain/model')
from mindy import mindy

def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))

#######################################################
# code starts here
subjlist = np.sort(os.listdir('/Users/hayoungsong/Documents/_postdoc/modelbrain/data/HCP/fmri/'))
subjlist = [folder for folder in subjlist if not folder.startswith('.DS')]
tasklist = ['tfMRI_MOVIE1_7T_AP', 'tfMRI_MOVIE2_7T_PA', 'tfMRI_MOVIE3_7T_PA', 'tfMRI_MOVIE4_7T_AP']
movielist = ['7T_MOVIE1_CC1_v2', '7T_MOVIE2_HO1_v2', '7T_MOVIE3_CC2_v2', '7T_MOVIE4_HO2_v2']

niter = 2500
batch_size = 300
nroi = 200
nFut = 5000

#######################################################
# sub_i, sub, ti, task = 0, subjlist[0], 0, tasklist[0]
for sub_i, sub in enumerate(subjlist):
    for ti, task in enumerate(tasklist):
        if os.path.exists('/Users/hayoungsong/Documents/_postdoc/modelbrain/output_attractors/HCP/input100/model_'+str(sub)+'_'+str(task)+'.npz')==False:
            print(str(sub_i+1)+'/'+str(len(subjlist))+': sub-'+str(sub)+', '+str(task))
            seed = int(sub) * 100 + ti
            torch.manual_seed(seed), np.random.seed(seed)

            ts = scipy.io.loadmat('/Users/hayoungsong/Documents/_postdoc/modelbrain/data/HCP/fmri/'+str(sub)+'/'+task+'.mat')['ts'].T
            ts = scipy.stats.zscore(ts, 1)
            input = np.load('/Users/hayoungsong/Documents/_postdoc/modelbrain/regressor_pc/HCP/dim100/'+movielist[ti]+'_input_pc.npy').T

            print('ts:    '+str(ts.shape))
            print('input: '+str(input.shape))

            # functional connectivity calculation
            fc_z = conv_r2z(np.corrcoef(ts))
            np.fill_diagonal(fc_z, 0)

            # model
            nT = ts.shape[1]
            nInput = input.shape[0]
            nParcel = ts.shape[0]
            # ******************************************
            if 'model' in locals(): del model
            model = mindy(nParcel, nInput)
            # model = torch.load('/Users/hayoungsong/Documents/_postdoc/modelbrain/output_attractors/HCP/input100/model_'+str(sub)+'_'+str(task)+'.pth', weights_only=False)

            iter_loss, iter_acc, iter_fcr = [], [], []
            for iter in range(niter):
                t_ = np.random.permutation(nT - batch_size)[0]
                y, yhat, J = model.forward(ts[:, t_:t_ + batch_size], input[:, t_:t_ + batch_size])
                model.update_weights(J)
            Xpred, pW, pD, pB = model.predict(ts, input)

            # prediction of next time step (accuracy)
            y = ts[:, 1:]
            y_pred = model.to_numpy(Xpred)
            ss_total = np.sum((y - np.mean(y)) ** 2)  # Total variance
            ss_residual = np.sum((y - y_pred) ** 2)  # Residual variance
            r_squared = 1 - (ss_residual / ss_total)
            print('rsq: ' + str(r_squared))

            # torch.save(model, '/Users/hayoungsong/Documents/_postdoc/modelbrain/output_attractors/HCP/input100/model_'+str(sub)+'_'+str(task)+'.pth')
            # *****************************************
            alpha = model.to_numpy(model.alpha)
            b = model.b
            W = model.to_numpy(model.W)
            Decay = model.to_numpy(model.Decay)
            beta = model.to_numpy(model.beta)

            xpred_time_start = np.zeros((nParcel, ts.shape[1]))
            xpred_time_end = np.zeros((10,nParcel, ts.shape[1]))
            term_beta = np.zeros((nParcel, ts.shape[1]))
            term_w_ = np.zeros((nParcel, ts.shape[1]))
            term_d_ = np.zeros((nParcel, ts.shape[1]))
            for t_ in range(ts.shape[1]):
                if np.mod(t_, 100)==0:
                    print('    '+str(t_)+' / '+str(ts.shape[1]))
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
                        term_beta[:, t_] = term_B
                        term_w_[:,t_] = term_W
                        term_d_[:,t_] = term_D
                        xpred_time_start[:,t_] = x_pred
                    if t>=nFut-10:
                        xpred_time_end[t-(nFut-10),:,t_] = x_pred

            fixed_points = np.zeros((nParcel, ts.shape[1])) + np.nan
            for t_ in range(ts.shape[1]):
                if np.all(np.abs(np.diff(xpred_time_end, axis=0))<1e-6):
                    fixed_points[:, t_] = xpred_time_end[-1,:,t_]
            print(str(len(np.where(np.isnan(fixed_points[0,:]))[0]))+ ' out of '+str(ts.shape[1])+' not converged to a fixed point')

            np.savez_compressed('/Users/hayoungsong/Documents/_postdoc/modelbrain/output_attractors/HCP/input100/model_'+str(sub)+'_'+str(task),
                                    pW=pW, pD=pD, pB=pB, rsq=r_squared, xpred_time_start=xpred_time_start, xpred_time_end=xpred_time_end,
                                    fixed_points=fixed_points, term_W = term_w_, term_D = term_d_, term_B = term_beta)
            print('')

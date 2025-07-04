import numpy as np
import scipy
import os

def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))

def cosine_angle(a, b):
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle_rad

def magnitude(a):
    return np.linalg.norm(a)

directory = '/set/directory' # ****** set directory *****
sublist = np.sort([f for f in os.listdir(directory+'/data/song2023elife/fMRI') if f.startswith('sub-')])
condlist = ['sitcomep1', 'sitcomep2', 'gradCPTface', 'gradCPTscene']

angle_wd_fp_cat, mag_wd_cat, angle_b_fp_cat, mag_b_cat = np.zeros((len(sublist), len(condlist)))+np.nan, np.zeros((len(sublist), len(condlist)))+np.nan, np.zeros((len(sublist), len(condlist)))+np.nan, np.zeros((len(sublist), len(condlist)))+np.nan
for ci, condition in enumerate(condlist):
    if condition in ['sitcomep1', 'sitcomep2']:
        behavior = scipy.io.loadmat(directory+'/data/song2023elife/beh/'+condition+'_beh.mat')['engagement_conv']
    elif condition in ['gradCPTface', 'gradCPTscene']:
        behavior = scipy.io.loadmat(directory+'/data/song2023elife/beh/'+condition+'_beh.mat')['rtvariability_conv']*(-1)

    for si, sub in enumerate(sublist):
        if condition=='sitcomep1' and sub=='sub-026': pass
        else:
            ts = scipy.io.loadmat(directory+'/data/song2023elife/fMRI/'+sub+'/'+condition+'.mat')['ts'].T
            ts = scipy.stats.zscore(ts, 1)
            if condition in ['sitcomep1', 'sitcomep2', 'gradCPTface']:
                input = np.load(directory+'/data/song2023elife/input/'+condition+'_input_pc.npy').T
            elif condition=='gradCPTscene':
                input = np.load(directory+'/data/song2023elife/input/'+condition+'_'+sub+'_input_pc.npy').T
            beh = behavior[:, si]

            if np.all(np.isnan(beh)): pass
            else:
                model_output = np.load(directory+'/output/'+condition+'_'+sub+'.npz')
                
                if np.any(np.isnan(model_output['fixed_points'])): print(str(sub)+' '+str(task))
                else:
                    term_WD = model_output['term_W'] + model_output['term_D']
                    term_B = model_output['term_B']
                    fp = model_output['fixed_points'] - ts

                    angle_wd_fp, angle_b_fp, mag_wd, mag_b = np.zeros((ts.shape[1],)), np.zeros((ts.shape[1],)), np.zeros((ts.shape[1],)), np.zeros((ts.shape[1],))
                    for t in range(ts.shape[1]):
                        angle_wd_fp[t] = cosine_angle(term_WD[:,t], fp[:,t])
                        angle_b_fp[t] = cosine_angle(term_B[:,t], fp[:,t])
                        mag_wd[t] = magnitude(term_WD[:,t])
                        mag_b[t] = magnitude(term_B[:,t])
                    angle_wd_fp_cat[si, ci] = conv_r2z(scipy.stats.spearmanr(angle_wd_fp, beh, nan_policy='omit')[0])
                    mag_wd_cat[si, ci] = conv_r2z(scipy.stats.spearmanr(mag_wd, beh, nan_policy='omit')[0])
                    angle_b_fp_cat[si, ci] = conv_r2z(scipy.stats.spearmanr(angle_b_fp, beh, nan_policy='omit')[0])
                    mag_b_cat[si, ci] = conv_r2z(scipy.stats.spearmanr(mag_b, beh, nan_policy='omit')[0])

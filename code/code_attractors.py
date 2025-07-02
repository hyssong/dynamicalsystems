import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def estimate_attractors(fixed_points):
    unique_fixed_points = []
    for t in range(fixed_points.shape[1]):
        if t == 0: unique_fixed_points.append(fixed_points[:,t])
        else:
            euc = []
            for u in unique_fixed_points:
                euc.append(np.linalg.norm(fixed_points[:,t]-u))
            if np.any(np.array(euc)<0.1): pass
            else: unique_fixed_points.append(fixed_points[:, t])

    fp_group = {}
    for g in range(len(unique_fixed_points)): fp_group[g] = []
    for t in range(fixed_points.shape[1]):
        if not np.all(np.isnan(fixed_points[:,t])):
            euc = np.zeros(len(unique_fixed_points),)
            for g in range(len(unique_fixed_points)):
                euc[g] = np.linalg.norm(unique_fixed_points[g]-fixed_points[:,t])
            fp_group[np.where(euc<0.1)[0][0]].append(fixed_points[:,t])
    attractors = np.zeros((fixed_points.shape[0],len(unique_fixed_points)))
    for g in range(len(unique_fixed_points)):
        attractors[:,g] = np.mean(np.array(fp_group[g]),0)

    attractors_id = np.zeros((fixed_points.shape[1],), dtype='int')
    for t in range(fixed_points.shape[1]):
        corrs = [np.corrcoef(fixed_points[:,t], attractors[:, i])[0, 1] for i in range(attractors.shape[1])]
        attractors_id[t] = np.argmax(corrs)
    return attractors, attractors_id


directory = '/set/directory' # ****** set directory *****
filelist = [f for f in os.listdir(directory+'/output') if f.endswith('.npz')]

# aggregate attractors of all runs
attractors_cat = []
nattractors = []
nRun = 0
for file in filelist:
    nRun=nRun+1
    fixed_points = np.load(directory+'/output/'+file)['fixed_points']
    if np.any(np.isnan(fixed_points)):
        nattractors.append(0)
    else:
        attractors, _ = estimate_attractors(fixed_points)
        nattractors.append(attractors.shape[1])
        if len(attractors_cat)==0: attractors_cat = attractors
        else: attractors_cat = np.concatenate((attractors_cat, attractors), 1)

# apply k-means clustering to find 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(attractors_cat.T)
clusters = kmeans.cluster_centers_.T

# load Margulies et al. gradients
gradients = scipy.io.loadmat(directory+'/data/gradientcoeff.mat')['gradientcoeff'][:,:2]

# find correspondence between the attractor clusters & top 2 gradients
coef_gradients_clusters = np.zeros((gradients.shape[1], clusters.shape[1]))
for i1 in range(gradients.shape[1]):
    for i2 in range(clusters.shape[1]):
        coef_gradients_clusters[i1,i2] = np.corrcoef(gradients[:,i1], clusters[:,i2])[0,1]

# visualize k-means clustering output in a 2D PC space
pca = PCA(n_components=2)
attractors_pc = pca.fit_transform(attractors_cat.T)
print(pca.explained_variance_ratio_)

fig, ax = plt.subplots(1, 1, figsize=(5.5,5))
for t in range(attractors_pc.shape[0]):
    if kmeans.labels_[t]==0: ax.scatter(attractors_pc[t, 1], attractors_pc[t, 0], color='#1f77b4', s=1)
    elif kmeans.labels_[t]==1: ax.scatter(attractors_pc[t, 1], attractors_pc[t, 0], color='#ff7f0e', s=1)
    elif kmeans.labels_[t]==2: ax.scatter(attractors_pc[t, 1], attractors_pc[t, 0], color='#2ca02c', s=1)
    elif kmeans.labels_[t]==3: ax.scatter(attractors_pc[t, 1], attractors_pc[t, 0], color='#d62728', s=1)
ax.set_xlabel("PC2"), ax.set_ylabel("PC1")

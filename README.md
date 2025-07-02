A dynamical systems model (known as MINDy, developed by [Singh et al., 2020](https://doi.org/10.1016/j.neuroimage.2020.117046) and [Chen et al., 2025](https://doi.org/10.1162/imag_a_00442)) that fits large-scale neural activity time series to estimate model parameters. Model parameters are used to find attractors and fluctuations in attractor landscape over time. Fluctuations in attractor landscape are related with dynamic measures of attentional states, collected by [Song et al., 2023](https://elifesciences.org/articles/85487).

**code**
- modelfit_winput.py: 


**data**
- song2023elife
  - slidingBeh.m : Generate group-average engagement timecourse (applies HRF convolution and sliding window to relate with fMRI data)
  - slidingBeh_surr.m : Generate phase-randomized behavioral timecourses for non-parametric permutation test
  - slidingFC.m : Generate time-resolved functional connectivity matrices from BOLD timeseries, using sliding window analysis
- gradientcoeff.mat: Cortical gradients found by [Margulies et al., 2016](https://neurovault.org/collections/1598/) summarized into 200-parcel [Schaefer](https://github.com/ThomasYeoLab/CBIG/tree/v0.14.3-Update_Yeo2011_Schaefer2018_labelname/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI) cortical atlas. 

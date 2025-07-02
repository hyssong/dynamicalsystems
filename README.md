A dynamical systems model (known as MINDy, developed by [Singh et al., 2020](https://doi.org/10.1016/j.neuroimage.2020.117046) and [Chen et al., 2025](https://doi.org/10.1162/imag_a_00442)) that fits large-scale neural activity time series to estimate model parameters. Model parameters are used to find attractors and fluctuations in attractor landscape over time. Fluctuations in attractor landscape are related with dynamic measures of attentional states, collected by [Song et al., 2023](https://elifesciences.org/articles/85487).

**code**
- modelfit_winput.py: imports fMRI and input time series to fit mindy model and estimate attractors using forward simulation of the model
- code_parameter.py: compare saved model parameters with descriptive statistics: parameter W with functional connectivity and parameter B with stimulus-to-brain encoding coefficients
- code_attractors.py: aggregate attractors estimated from all runs and compare with cortical gradients
- code_attention.py: calculate angle and magnitude of neural dynamics toward attractors and relate those with dynamic attention measures 

**data**
- song2023elife
  - beh: processed behavioral time series of 27 participants. further data description can be found in [/neuraldynamics/](https://github.com/hyssong/neuraldynamics).
  - fmri: processed fMRI data (200 parcel x time) of 27 participants. further data description can be found in [/neuraldynamics/](https://github.com/hyssong/neuraldynamics).
  - input: input embedding time series, including visual, auditory, and semantic features of the stimuli. stimuli presentation was randomized for each individual in "gradCPTscene" run, but the same stimuli was used for other conditions.
- gradientcoeff.mat: Cortical gradients found by [Margulies et al., 2016](https://neurovault.org/collections/1598/) summarized into 200-parcel [Schaefer](https://github.com/ThomasYeoLab/CBIG/tree/v0.14.3-Update_Yeo2011_Schaefer2018_labelname/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI) cortical atlas. 

**model**
- mindy.py: a simplified pytorch implementation of the MINDy model. this function can be used to fit any z-scored neural activity data of (neural unit x time). the model can be fit with or without inputs.

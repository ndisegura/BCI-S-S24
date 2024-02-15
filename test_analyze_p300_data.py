# -*- coding: utf-8 -*-
"""
andres Segura & Skyler Younger
BME 6770: BCI's Lab 02
Dr. David Jangraw
2/12/2024

"""


import os
# Inport the necessry modules
import sys
import load_p300_data
import plot_p300_erp
import matplotlib.pyplot as plt
import numpy as np
import analyze_p300_data

#Make sure relative path work
cwd=os.getcwd()
sys.path.insert(0,f"{cwd}\course_software\BCIs-S24-main\\")

#Close previosly drawn plots
plt.close('all')

#Build data file string
data_directory='course_software/P300Data/'
subject=4
data_file=f'{cwd}{data_directory}s{subject}.mat'

#%% Part A: Load and Epoch the Data

eeg_epochs, eeg_epochs_target, eeg_epochs_nontarget, erp_times=analyze_p300_data.load_and_epoch_data(subject, data_directory)

#%% Part B: Calculate and Plot Parametric Confidence Intervals

analyze_p300_data.calculate_and_plot_confidence_intervals(eeg_epochs_target, eeg_epochs_nontarget,erp_times)

#%% Part C: Bootstrap P values

#analyze_p300_data.resample_data(eeg_epochs_target[:,0,0], eeg_epochs_target.shape[0])
#resampled_target_epochs=analyze_p300_data.resample_data(eeg_epochs_target, 1000)
analyze_p300_data.bootstrap_eeg_erp (eeg_epochs, eeg_epochs_target, eeg_epochs_nontarget)
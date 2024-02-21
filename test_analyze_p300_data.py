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
import math
import scipy as sci
import plot_topo

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
bootstrapped_distribution=analyze_p300_data.bootstrap_eeg_erp(eeg_epochs, eeg_epochs_target, eeg_epochs_nontarget,1000)
epoch_diff_p_values = analyze_p300_data.find_sample_p_value(bootstrapped_distribution, eeg_epochs_target, eeg_epochs_nontarget, erp_times)


#%% Part D: Multiple comparisons

significant_samples,significant_plot_samples, corrected_p_values, is_significant_int = analyze_p300_data.p_value_fdr_correction(epoch_diff_p_values)

analyze_p300_data.plot_significant_p_values(eeg_epochs_target, eeg_epochs_nontarget, significant_plot_samples, erp_times)

#%% Part E: Evaluate across subjects
#Define constants
First_Subject_index=3
Last_Subject_Index=10
data_path=cwd+data_directory+"s"
significant_subject_count,erp_times,combined_erp_target_mean,combined_erp_nontarget_mean=analyze_p300_data.analyze_across_subjects(First_Subject_index,Last_Subject_Index,data_directory)

analyze_p300_data.plot_significance_across_subjects(significant_subject_count,erp_times)

#%% Part F: Plot Spatial Map
channel_names=['PO7','PO8','Fz','P3','P4','Oz','P6','Cz']
analyze_p300_data.get_p3b_range(erp_times,combined_erp_target_mean,combined_erp_nontarget_mean)
plot_topo.plot_topo(channel_names,combined_erp_nontarget_mean)
#a=plot_topo.get_channel_names()



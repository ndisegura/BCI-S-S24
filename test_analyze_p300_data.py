# -*- coding: utf-8 -*-
"""
andres Segura & Skyler Younger
BME 6770: BCI's Lab 02
Dr. David Jangraw
2/12/2024

This script intends to explore the results of a
P300 based BCI 2D experiment (Guger et al. 2009) in which subjects were asked to spell
a word based on their attention re-orienting response (P300 ERP). The scrips calls functions defined
in the analyze_p300_data.py file to produce mean and confidence Intervals plots as well as 
identifuing statistical significant difference between target vs nontarget events for each EEG samples 
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
from course_software import plot_topo

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

analyze_p300_data.calculate_and_plot_confidence_intervals(eeg_epochs_target, eeg_epochs_nontarget,erp_times, subject)

#%% Part C: Bootstrap P values

bootstrapped_distribution=analyze_p300_data.bootstrap_eeg_erp(eeg_epochs, eeg_epochs_target, eeg_epochs_nontarget,500)
epoch_diff_p_values = analyze_p300_data.find_sample_p_value(bootstrapped_distribution, eeg_epochs_target, eeg_epochs_nontarget, erp_times)


#%% Part D: Multiple comparisons

significant_plot_samples, corrected_p_values, is_significant_int = analyze_p300_data.p_value_fdr_correction(epoch_diff_p_values)

analyze_p300_data.plot_significant_p_values(eeg_epochs_target, eeg_epochs_nontarget, significant_plot_samples, erp_times,subject)

#%% Part E: Evaluate across subjects
#Define constants
first_subject_index=3
last_subject_index=10

significant_subject_count,erp_times,combined_erp_target_median,combined_erp_nontarget_median,subjects_target_median,subjects_nontarget_median=analyze_p300_data.analyze_across_subjects(first_subject_index,last_subject_index,data_directory)

analyze_p300_data.plot_significance_across_subjects(significant_subject_count,erp_times)

#%% Part F: Plot Spatial Map
channel_names=['P4','PO8','P3','PO7','Oz','Fz','Cz','Pz']
#Subject 3
p3b_target_range,p3b_nontarget_range=analyze_p300_data.get_erp_range(erp_times,subjects_target_median[0,:,:],subjects_nontarget_median[0,:,:], 0.25, 0.5)
n200_target_range,n200_nontarget_range=analyze_p300_data.get_erp_range(erp_times,subjects_target_median[0,:,:],subjects_nontarget_median[0,:,:], 0.2, 0.350)

plot_topo.plot_topo(channel_names,p3b_target_range)
plot_topo.plot_topo(channel_names,p3b_nontarget_range)
plot_topo.plot_topo(channel_names,n200_target_range)
plot_topo.plot_topo(channel_names,n200_nontarget_range)


#Subject 8
p3b_target_range,p3b_nontarget_range=analyze_p300_data.get_erp_range(erp_times,subjects_target_median[5,:,:],subjects_nontarget_median[0,:,:], 0.25, 0.5)
n200_target_range,n200_nontarget_range=analyze_p300_data.get_erp_range(erp_times,subjects_target_median[5,:,:],subjects_nontarget_median[0,:,:], 0.2, 0.350)

plot_topo.plot_topo(channel_names,p3b_target_range)
plot_topo.plot_topo(channel_names,p3b_nontarget_range)
plot_topo.plot_topo(channel_names,n200_target_range)
plot_topo.plot_topo(channel_names,n200_nontarget_range)




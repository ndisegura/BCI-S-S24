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


#Build data file string
data_directory='course_software/P300Data/'
subject=3
data_file=f'{cwd}{data_directory}s{subject}.mat'

#%% Part A: Load and Epoch the Data

eeg_epochs_target, eeg_epochs_nontarget=analyze_p300_data.load_and_epoch_data(subject, data_directory)

#%% Part B: Calculate and Plot Parametric Confidence Intervals

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

#Make sure relative path work
cwd=os.getcwd()
sys.path.insert(0,f"{cwd}\course_software\BCIs-S24-main\\")


#Build data file string
data_directory='course_software/P300Data/'
subject=3
data_file=f'{cwd}{data_directory}s{subject}.mat'


def load_and_epoch_data(subject, data_directory):
    
    #Load Training Data
    [eeg_time,eeg_data,rowcol_id,is_target]=load_p300_data.load_training_eeg(subject=3, data_directory=data_directory);
    #Find Event Samples
    event_sample, is_target_event=plot_p300_erp.get_events(rowcol_id, is_target)
    #Extract the Epochs
    eeg_epochs, erp_times = plot_p300_erp.epoch_data(eeg_time,eeg_data,event_sample)
    #Find Target and Non-Target Epochs
    eeg_epochs_target, eeg_epochs_nontarget = plot_p300_erp.get_erps(eeg_epochs, is_target_event)
    #Visualize ERPs
    plot_p300_erp.plot_erps(eeg_epochs_target, eeg_epochs_nontarget, erp_times)
    plt.show()

    
    return eeg_epochs_target, eeg_epochs_nontarget
    
"""
Lincoln Lewis & Andres Segura
BME 6770: BCI's Lab 02
Dr. David Jangraw
2/7/2024

This script intends to explore the results of a
P300 based BCI 2D experiment (Guger et al. 2009) in which subjects were asked to spell
a word based on their attention re-orienting response (P300 ERP)

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


#%% Cell1:  Load Trainning Data
#Load Data
[eeg_time,eeg_data,rowcol_id,is_target]=load_p300_data.load_training_eeg(subject=3, data_directory=data_directory);


#%% Cell2 Find Event Samples

event_sample, is_target_event=plot_p300_erp.get_events(rowcol_id, is_target)

#%%
'''
Part 3: Extract the epochs
'''
fs = 1 / (eeg_time[1]-eeg_time[0])

# based on my (lincolns) load_p300 data, each epoch of data should be 33 seconds in length.
eeg_epochs, erp_times = plot_p300_erp.epoch_data(eeg_time,eeg_data,event_sample)

#%%
'''
Part 4: Calculate the ERP's 
'''

eeg_epochs_target, eeg_epochs_nontarget = plot_p300_erp.get_erps(eeg_epochs, is_target_event)

#%%
'''
Part 5: Plot the ERP's
'''
plot_p300_erp.plot_erps(eeg_epochs_target, eeg_epochs_nontarget, erp_times)


plt.show()

#%%
'''
Part 6: Analysis + multiple plots 
'''
# plot multiiple subjects to compare for analysis
for subject_index in range(4):
    #Update File names
    subject=subject_index+1
    data_file=f'{cwd}{data_directory}s{subject}.mat'
    #Call the different functions to plot data
    #Load the data
    [eeg_time,eeg_data,rowcol_id,is_target]=load_p300_data.load_training_eeg(subject, data_directory=data_directory);
    #Find Event samples
    event_sample, is_target_event=plot_p300_erp.get_events(rowcol_id, is_target)
    #Epoch Data
    eeg_epochs, erp_times = plot_p300_erp.epoch_data(eeg_time,eeg_data,event_sample)
    #Compute ERP
    eeg_epochs_target, eeg_epochs_nontarget = plot_p300_erp.get_erps(eeg_epochs, is_target_event)
    #Plot ERPs
    plot_p300_erp.plot_erps(eeg_epochs_target, eeg_epochs_nontarget, erp_times)
    
'''
1. Why do we see repeated up-and-down patterns on many of the channels?
    They are most pronounced on the target event compared to the non-target events which appear to be mostly oscillatory signals.
    The up-and-down patters are located mostly shorty after time zero. They must be action Potentials

2. Why are they more pronounced in some channels but not others?
    The action potential magnitudes must be related to the electrode or electrodes situated closest to the part of the brain with the
    most activity at that moment.

3. Why does the voltage on some of the channels have a positive peak around half a 
    second after a target flash?
    There appears to be a series of excitatory post synaptic potentials and inhibitory post synaptic potentials
    before the target event. When several excitory post-synaptic potential "pile-up" an action potential occurs.

4. Which EEG channels (e.g., Cz) do you think these (the ones described in the last question) 
    might be and why? 
    We believe those signals might below to channels O1, O2 and POz which belong to the occipital cortex area. The P300
    speller training and operation is visual in nature, and the region  of the brain responsible for vision and
    perception corresponds to the occipital cortex.
    

'''

    
    
    
    



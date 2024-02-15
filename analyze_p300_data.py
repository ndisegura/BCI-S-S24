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
import math
import random

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

    
    return eeg_epochs,eeg_epochs_target, eeg_epochs_nontarget, erp_times
    
def calculate_and_plot_confidence_intervals(eeg_epochs_target, eeg_epochs_nontarget, erp_times):
    
    #Compute the Mean for Target and Non-targets
    target_mean=np.mean(eeg_epochs_target, axis=0)
    nontarget_mean=np.mean(eeg_epochs_nontarget, axis=0)
    
    #Compute the standard deviation and std error
    target_std=np.std(eeg_epochs_target, axis=0)/math.sqrt(eeg_epochs_target.shape[0]) #Divide by number of trials
    #target_std=np.std(eeg_epochs_target, axis=0)#I believe np.std aready divives by n
    #nontarget_std=np.std(eeg_epochs_nontarget, axis=0) 
    nontarget_std=np.std(eeg_epochs_nontarget, axis=0)/ math.sqrt(eeg_epochs_nontarget.shape[0]) #Divide by number of trials
    
    #Plot the results
    fig, axs = plt.subplots(3,3)
    
    
    for plot_index, ax in enumerate(axs.flatten()):
        if plot_index ==8 :
            ax.set_visible(False) #This channel doesn't exist
        else:
            ax.plot(erp_times, target_mean[plot_index,:], 'b', lw=1,label='target')              # Plot the ERP of condition A
            ax.plot(erp_times, target_mean[plot_index,:] + 2 * target_std[plot_index,:], 'b:', lw=1)  # ... and include the upper CI
            ax.plot(erp_times, target_mean[plot_index,:] - 2 * target_std[plot_index,:], 'b:', lw=1)  # ... and the lower CI
            ax.plot(erp_times, nontarget_mean[plot_index,:], 'm', lw=1,label='non-target')              # Plot the ERP of condition A
            ax.plot(erp_times, nontarget_mean[plot_index,:] + 2 * target_std[plot_index,:], 'm:', lw=1)  # ... and include the upper CI
            ax.plot(erp_times, nontarget_mean[plot_index,:] - 2 * target_std[plot_index,:], 'm:', lw=1)  # ... and the lower CI
            ax.set_title(f'Channel {plot_index}')
            ax.set_xlabel('Time from flash onset (s)')
            ax.set_ylabel('Voltage ($\mu$ V)')
        
            ax.legend()
            ax.grid()
            ax.axvline(x=0, color='black', linestyle='--')
            ax.axhline(y=0, color='black', linestyle='--')
    plt.tight_layout()
    fig.suptitle(' P300 ERPs 95% Confidence Intervals ')
    fig                                    # ... and show the plot
    plt.show()
    
def resample_data(input_data, number_iterations):
    #Declare numpy vector to store the re-sampled data
    resampled_data=np.zeros((number_iterations,input_data.shape[1],input_data.shape[2]))
    
    #nested loop to iterate through pages and channels and select random samples
    for iteration_index in range(number_iterations):
    
        for channels_index in range(input_data.shape[1]):
            
            for sample_index in range(input_data.shape[2]):
            
                resampled_data[iteration_index,channels_index,sample_index]=random.choice(input_data[:,channels_index,sample_index])
        
    return np.mean(resampled_data, axis=0)

def bootstrap_eeg_erp (eeg_epochs, eeg_epochs_target, eeg_epochs_nontarget):
    
    #resample Target
    resampled_mean_epoch_target=resample_data(eeg_epochs,eeg_epochs_target.shape[0])
    resampled_mean_epoch_nontarget=resample_data(eeg_epochs,eeg_epochs_nontarget.shape[0])
    
    #Compute the stat
    null_hypothesis_stat=np.absolute(resampled_mean_epoch_target-resampled_mean_epoch_nontarget)
    
    pass


    
    
    
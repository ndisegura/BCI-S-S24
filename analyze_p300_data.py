# -*- coding: utf-8 -*-
"""
andres Segura & Skyler Younger
BME 6770: BCI's Lab 02
Dr. David Jangraw
2/12/2024

"""


import os
# Import the necessry modules
import sys
import load_p300_data
import plot_p300_erp
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import scipy as sci
from mne.stats import fdr_correction

#Make sure relative path work
cwd=os.getcwd()
sys.path.insert(0,f"{cwd}\course_software\BCIs-S24-main\\")


#Build data file string
data_directory='course_software/P300Data/'
# subject=3
# data_file=f'{cwd}{data_directory}s{subject}.mat'


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
    # plot_p300_erp.plot_erps(eeg_epochs_target, eeg_epochs_nontarget, erp_times)
    # plt.show()

    
    return eeg_epochs,eeg_epochs_target, eeg_epochs_nontarget, erp_times
    
def calculate_and_plot_confidence_intervals(eeg_epochs_target, eeg_epochs_nontarget, erp_times, subject = 3):
    
    #Compute the Mean for Target and Non-targets
    target_mean=np.mean(eeg_epochs_target, axis=0)
    nontarget_mean=np.mean(eeg_epochs_nontarget, axis=0)
    
    #Compute the standard deviation and std error
    target_std=np.std(eeg_epochs_target, axis=0)/math.sqrt(eeg_epochs_target.shape[0]) #Divide by number of trials
    #target_std=np.std(eeg_epochs_target, axis=0)#I believe np.std aready divives by n
    #nontarget_std=np.std(eeg_epochs_nontarget, axis=0) 
    nontarget_std=np.std(eeg_epochs_nontarget, axis=0)/ math.sqrt(eeg_epochs_nontarget.shape[0]) #Divide by number of trials
    
    #Plot the results
    fig, axs = plt.subplots(3,3, figsize=(9,8))
    
    
    for plot_index, ax in enumerate(axs.flatten()):
        if plot_index == 8:
            if eeg_epochs_target.shape[0] == 8 :
                ax.set_visible(False) #This channel doesn't exist
        else:
            target_lower_ci = target_mean[plot_index,:] - 2 * target_std[plot_index,:]
            target_upper_ci = target_mean[plot_index,:] + 2 * target_std[plot_index,:]
            nontarget_lower_ci = nontarget_mean[plot_index,:] - 2 * nontarget_std[plot_index,:]
            nontarget_upper_ci = nontarget_mean[plot_index,:] + 2 * nontarget_std[plot_index,:]
            
            ax.plot(erp_times, target_mean[plot_index,:], 'b', lw=1,label='target')              # Plot the ERP of condition A
            ax.fill_between(erp_times,target_lower_ci,target_upper_ci)
            ax.plot(erp_times, nontarget_mean[plot_index,:], 'm', lw=1,label='non-target')              # Plot the ERP of condition A
            ax.fill_between(erp_times,nontarget_lower_ci,nontarget_upper_ci)
            ax.set_title(f'Channel {plot_index}')
            ax.set_xlabel('Time from flash onset (s)')
            ax.set_ylabel('Voltage ($\mu$ V)')
        
            ax.legend()
            ax.grid()
            ax.axvline(x=0, color='black', linestyle='--')
            ax.axhline(y=0, color='black', linestyle='--')
    fig.suptitle(f'Subject {subject} P300 ERPs 95% Confidence Intervals ')
    plt.tight_layout()
    fig                                    # ... and show the plot
    plt.show()
    
def resample_data(input_data, number_iterations):
    #Declare numpy vector to store the re-sampled data
    resampled_data=np.zeros((number_iterations,input_data.shape[1],input_data.shape[2]))
    
    ntrials=input_data.shape[0]
    size=number_iterations
    i = np.random.randint(ntrials, size=size)    # ... draw random trials,
    resampled_data=input_data[i]
    
    #nested loop to iterate through pages and channels and select random samples
    # for iteration_index in range(number_iterations):
    
    #     for channels_index in range(input_data.shape[1]):
            
    #         for sample_index in range(input_data.shape[2]):
            
    #             resampled_data[iteration_index,channels_index,sample_index]=random.choice(input_data[:,channels_index,sample_index])
        
    return np.mean(resampled_data, axis=0)

def bootstrap_eeg_erp (eeg_epochs, eeg_epochs_target, eeg_epochs_nontarget,bootstrap_count):
    
    bootstrapped_distribution=np.zeros([bootstrap_count,eeg_epochs.shape[1],eeg_epochs.shape[2]])
    for bootstrap_index in range(bootstrap_count):
        print(f'Loop count:{bootstrap_index}')
        #resample Target
        resampled_mean_epoch_target=resample_data(eeg_epochs,eeg_epochs_target.shape[0])
        resampled_mean_epoch_nontarget=resample_data(eeg_epochs,eeg_epochs_nontarget.shape[0])
        
        #Compute the stat
        null_hypothesis_stat=np.absolute(resampled_mean_epoch_target-resampled_mean_epoch_nontarget)
        #Build the new distribution
        bootstrapped_distribution[bootstrap_index,:,:]=null_hypothesis_stat
    return bootstrapped_distribution


def find_sample_p_value(bootstrapped_distribution, eeg_epochs_target, eeg_epochs_nontarget, erp_times):
    # Find sample size
    bootstrapped_sample_size = bootstrapped_distribution.shape[0]
    
    # Find the absolute value difference of each sample
    absolute_sample_diff = np.absolute(np.mean(eeg_epochs_target,axis=0) - np.mean(eeg_epochs_nontarget,axis=0))
    
    # Create empty array for p values at each time point and an boolean array 
    # to determine if each p value is significant
    epoch_diff_p_values = np.zeros([bootstrapped_distribution.shape[1],bootstrapped_distribution.shape[2]])
    
    
    # Create list used to sum number of time absolute_sample_diff is greater
    # than bootstrapped samples at each time point
    is_greater = [] 
    
    for channel_index in range(bootstrapped_distribution.shape[1]):
        # print(channel_index)
        for sample_index in range(bootstrapped_distribution.shape[2]):
            
            # Determine how many bootstrapped samples are smaller than the 
            # mean absolute value difference
            for bootstrap_index in range(bootstrapped_distribution.shape[0]):
                
                if absolute_sample_diff[channel_index,sample_index]<=bootstrapped_distribution[bootstrap_index,channel_index,sample_index]:
                    is_greater.append(0)
                elif absolute_sample_diff[channel_index,sample_index]>bootstrapped_distribution[bootstrap_index,channel_index,sample_index]:
                    is_greater.append(1)
    
            # Find p value for current sample
            sum_greater = sum(is_greater)
            is_greater = []
            # Calculate p value based on the number of bootstrapped samples which
            # are smaller than the target sample
            p_value = (bootstrapped_sample_size - sum_greater) / bootstrapped_sample_size
            if p_value == 0:
                p_value = 1/bootstrapped_sample_size
            
            epoch_diff_p_values[channel_index,sample_index] = p_value
            
    
    # Uncomment code below to create a graph showing where we expect the 
    # sample mean to be significantly different than the bootstrapped mean!
    
    # EEGa = absolute_sample_diff[0,:]
    # ERP0 = bootstrapped_distribution
    # ERP0.sort(axis=0)         # Sort each column of the resampled ERP
    # N = len(ERP0)             # Define the number of samples
    # ciL = ERP0[int(0.025*N),0,:]  # Determine the lower CI
    # ciU = ERP0[int(0.975*N),0,:]  # ... and the upper CI
    # # mnA = EEGa.mean(0)        # Determine the ERP for condition A
    # plt.plot(erp_times, EEGa, 'k', lw=3)   # ... and plot it
    # plt.plot(erp_times, ciL, 'k:')        # ... and plot the lower CI
    # plt.plot(erp_times, ciU, 'k:')        # ... and the upper CI
    # plt.hlines(1, 0, 1, 'b')      # plot a horizontal line at 0
    #                           # ... and label the axes
    # plt.title('ERP of condition A with bootstrap confidence intervals')  # We define this function above!
    return epoch_diff_p_values


def p_value_fdr_correction(epoch_diff_p_values, alpha = 0.05):
    
    significant_samples, corrected_p_values = fdr_correction(epoch_diff_p_values, alpha)
    
    significant_plot_samples = np.where(significant_samples == True, 0, None)
    is_significant_int = significant_samples = np.where(significant_samples == True, 1, 0)
    # significant_samples = np.zeros([corrected_p_values.shape[0],corrected_p_values.shape[1]])
    # for channel_index in range(corrected_p_values.shape[0]):
    #     # print(channel_index)
    #    for sample_index in range(corrected_p_values.shape[1]):
    #        if corrected_p_values[channel_index,sample_index] <= alpha:
    #            significant_samples[channel_index,sample_index] = 1
    return significant_plot_samples, corrected_p_values, is_significant_int
    
    
def plot_significant_p_values(eeg_epochs_target, eeg_epochs_nontarget, significant_plot_samples, erp_times, subject = 3,save_location='./'):
    
    #Compute the Mean for Target and Non-targets
    target_mean=np.mean(eeg_epochs_target, axis=0)
    nontarget_mean=np.mean(eeg_epochs_nontarget, axis=0)
    
    #Compute the standard deviation and std error
    target_std=np.std(eeg_epochs_target, axis=0)/math.sqrt(eeg_epochs_target.shape[0]) #Divide by number of trials
    #target_std=np.std(eeg_epochs_target, axis=0)#I believe np.std aready divives by n
    #nontarget_std=np.std(eeg_epochs_nontarget, axis=0) 
    nontarget_std=np.std(eeg_epochs_nontarget, axis=0)/ math.sqrt(eeg_epochs_nontarget.shape[0]) #Divide by number of trials
       
    
    #Plot the results
    fig, axs = plt.subplots(3,3, figsize=(9,8))
    
    
    for plot_index, ax in enumerate(axs.flatten()):
        if plot_index == 8:
            if significant_plot_samples.shape[0] == 8 :
                ax.set_visible(False) #This channel doesn't exist
        else:
            # Plot target
            ax.plot(erp_times, target_mean[plot_index,:], 'b', lw=1,label='target')
            target_lower_ci = target_mean[plot_index,:] - 2 * target_std[plot_index,:]
            target_upper_ci = target_mean[plot_index,:] + 2 * target_std[plot_index,:]
            ax.fill_between(erp_times,target_lower_ci,target_upper_ci)
            # Plot nontarget
            nontarget_lower_ci = nontarget_mean[plot_index,:] - 2 * nontarget_std[plot_index,:]
            nontarget_upper_ci = nontarget_mean[plot_index,:] + 2 * nontarget_std[plot_index,:]
            ax.plot(erp_times, nontarget_mean[plot_index,:], 'm', lw=1,label='non-target')
            ax.fill_between(erp_times,nontarget_lower_ci,nontarget_upper_ci)
            # Plot significant values
            ax.plot(erp_times, significant_plot_samples[plot_index,:], color = 'black', marker = 'o', ms = 3.5, mfc = 'purple', lw=0,label='significant') # Plot the ERP of condition A
            ax.set_title(f'Channel {plot_index}')
            ax.set_xlabel('Time from flash onset (s)')
            ax.set_ylabel('Voltage ($\mu$ V)')
        
            ax.legend(loc = 'upper left', fontsize=7)
            ax.grid()
            ax.axvline(x=0, color='black', linestyle='--')
            ax.axhline(y=0, color='black', linestyle='--')
    fig.suptitle(f'Subject {subject} P300 ERPs 95% Confidence Intervals ')
    plt.tight_layout()
    fig                                    # ... and show the plot
    plt.show()
    plt.savefig(f"{save_location}/P300_S{subject}_erps_significance.png")
    
def analyze_across_subjects(first_subject_index,last_subject_index,data_directory, array_shape=(8,384)):
    significant_subject_count=np.zeros(array_shape)
    subjects_target_mean=np.zeros((last_subject_index-first_subject_index+1,array_shape[0],array_shape[1]))
    subjects_nontarget_mean=np.zeros((last_subject_index-first_subject_index+1,array_shape[0],array_shape[1]))
    for subject_index, subject_id in enumerate(range(first_subject_index,last_subject_index+1)):
        print(f'Subject Index:{subject_index}')
        #Load and Epoch subject data
        eeg_epochs, eeg_epochs_target, eeg_epochs_nontarget, erp_times=load_and_epoch_data(subject_id, data_directory)
        #Compute Bootstrapped Distribution and p-values
        bootstrapped_distribution=bootstrap_eeg_erp(eeg_epochs, eeg_epochs_target, eeg_epochs_nontarget,3000)
        epoch_diff_p_values = find_sample_p_value(bootstrapped_distribution, eeg_epochs_target, eeg_epochs_nontarget, erp_times)
        #Compute FDR Correctec P-values
        significant_plot_samples, corrected_p_values, is_significant_int = p_value_fdr_correction(epoch_diff_p_values)
        #Plot and save each subject's ERPs
        plot_significant_p_values(eeg_epochs_target, eeg_epochs_nontarget, significant_plot_samples, erp_times)
        #Accumulate number of subject that pass the significant threshold
        significant_subject_count=significant_subject_count+is_significant_int
        
        #Compute the Mean for Target and Non-targets
        subjects_target_mean[subject_index,:,:]=np.mean(eeg_epochs_target, axis=0)
        subjects_nontarget_mean[subject_index,:,:]=np.mean(eeg_epochs_nontarget, axis=0)
        
    combined_erp_target_mean=np.mean(subjects_target_mean,axis=0)
    combined_erp_nontarget_mean=np.mean(subjects_nontarget_mean,axis=0)
    return significant_subject_count,erp_times,combined_erp_target_mean,combined_erp_nontarget_mean

def plot_significance_across_subjects(significant_subject_count,erp_times,save_location='./'):
    #Plot the results
    fig, axs = plt.subplots(3,3, figsize=(9,8))
    
    
    for plot_index, ax in enumerate(axs.flatten()):
        if plot_index == 8:
            if significant_subject_count.shape[0] == 8 :
                ax.set_visible(False) #This channel doesn't exist
        else:
            # Plot target
            ax.plot(erp_times, significant_subject_count[plot_index,:], 'b', lw=1,label='Significant Samples')
            ax.set_title(f'Channel {plot_index}')
            ax.set_xlabel('Time from flash onset (s)')
            ax.set_ylabel('Significant Subjects')
        
            ax.legend(loc = 'upper left', fontsize=7)
            ax.grid()
            ax.axvline(x=0, color='black', linestyle='--')
            ax.axhline(y=0, color='black', linestyle='--')
    fig.suptitle(' Significant Samples Across Subjects ')
    plt.tight_layout()
    
    fig                                    # ... and show the plot
    plt.show()
    plt.savefig(f'{save_location}/significant_subjects.png')
    
    
def get_p3b_range(erp_times,combined_erp_target_mean,combined_erp_nontarget_mean, start_time = 0.25, end_time = 0.5):
    
    is_p3b_range = np.zeros(erp_times.shape)
    is_p3b_range=np.where(((erp_times>=start_time)&(erp_times<=end_time)), 1,0)
    p3b_target_range=combined_erp_target_mean[:,np.where(is_p3b_range)]
    p3b_target_range=np.squeeze(p3b_target_range,axis=1)
    p3b_nontarget_range=combined_erp_nontarget_mean[:,np.where(is_p3b_range)]
    p3b_nontarget_range=np.squeeze(p3b_nontarget_range,axis=1)
    
    return p3b_target_range,p3b_nontarget_range
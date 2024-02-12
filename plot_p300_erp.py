# -*- coding: utf-8 -*-
"""
Lincoln Lewis * andres Segura
BME 6770: BCI's Lab 02
Dr. David Jangraw
2/7/2024

This module provides functions for processing and analyzing raw EEG data from the subjects in the Guger et al. 2009 paper.

The get_events function extracts event samples and corresponding target information from rowcol_id and is_target arrays.
The epoch_data function epochs the EEG data based on the event samples, and a start and end time for the epochs.
The get_erps function separates the epoched EEG data into target and non-target trials.
The plot_erps function plots the ERPs for target and non-target trials.

"""

import matplotlib.pyplot as plt
import numpy as np


def get_events(rowcol_id, is_target):
    """
    Extracts event samples and corresponding target information from rowcol_id and is_target arrays.
    Parameters:
    - rowcol_id (numpy.ndarray): Array containing row and column IDs, of integer type, of size n x 1
    - is_target (numpy.ndarray): Boolean array indicating whether or not a target appears, of size n x 1

    Returns:
    - event_sample (numpy.ndarray): Array containing the indexes of event samples, of size n x 1, and ints
    - is_target_event (numpy.ndarray): Boolean array indicating whether the event is a target, of size n x 1, only boolean values (T/F)
    """
    # take the difference so that we can see when the rowcol_id goes positive
    rowcol_id_diff=np.diff(rowcol_id)
    # now get all of the indexes/events when the rowcol_id_diff is greater than zero
    event_sample=np.where(rowcol_id_diff>0)
    # add 1 to each event sample so we know exactly where the event is
    event_sample=event_sample[0]+1
    # now index is_target to get when we want our epochs to be
    is_target_event=is_target[event_sample]

    
    return event_sample, is_target_event

def epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time = -0.5, epoch_end_time = 1):
    """
    Epochs each channel based off of when events happen and when
    the start time and end time is
    The sampling freqeuncy is first calculated based off the difference in time between two successive samples,
    simply just taking the first and second index.
    # of samples before and after are then calculated, which we use as our indexing for
    creating our 3D array calls eeg_epochs

    Parameters:
    - eeg_time (numpy.ndarray): Array containing EEG time data in seconds, size n x 1 and floats
    - eeg_data (numpy.ndarray): Array containing EEG data of size n x 1 and floats
    - event_sample (numpy.ndarray): Array containing the indexes of event samples, size n x 1 and integers
    - epoch_start_time (float): Start time of each epoch in seconds (default is -0.5).
    - epoch_end_time (float): End time of each epoch in seconds (default is 1).

    Returns:
    - eeg_epochs (numpy.ndarray): Array containing the epoched EEG data, of size nxnxn (3D array) Where axis 0 represents the epochs,
         axis 1 is each channel per epoch, and axis 2 is a 1D array of voltage (floats)
    - erp_times (numpy.ndarray): Array containing the time in seconds of each sample in the epoch relative to the event onset, for each channel. Of size n x n of floats
    """
    # first we will find the sampling frequency, in Hz
    # or samples per second
    # this sampling frequency should be ~256Hz according to Guger et al. 2009 paper
    fs = 1 / (eeg_time[1]-eeg_time[0])

    # now define the # of samples before the event basefd on start time
    samples_before_event = int(fs*np.abs(epoch_start_time))
    # now after the event
    samples_after_event = int(fs*np.abs(epoch_end_time))
    # now create an array of all of the start times
    target_times = eeg_time[event_sample]
    print(len(event_sample))
    num_samples = samples_after_event + samples_before_event
    eeg_epochs = np.empty((len(event_sample), np.shape(eeg_data)[0], num_samples))
    erp_times = np.arange(epoch_start_time, epoch_end_time, 1 / fs)

    # Iterate over each event sample
    for index, time in enumerate(event_sample):
        # Calculate the start and end index for the samples around the event time
        start_index = time - samples_before_event
        end_index = time + samples_after_event

        # Extract the epoch data for all channels
        epoch_data = eeg_data[:, start_index:end_index]

        # Assign the epoch data to the corresponding index in eeg_epochs
        eeg_epochs[index] = epoch_data

    # return them finally
    return eeg_epochs, erp_times
    


def get_erps(eeg_epochs, is_target_event):
    """
    Separates the epoched EEG data into target and non-target trials
    based off if an event occured and all of the epochs we want to index
    This just uses simply boolean indexing to get the non target and target

    Parameters:
    - eeg_epochs (numpy.ndarray): Array containing the epoched EEG data, of size nxnxn (3D array), of int x int x floats
    - is_target_event (numpy.ndarray): Boolean array indicating whether the event is a target.

    Returns:
    - eeg_epochs_target (numpy.ndarray): Array containing epoched EEG data for target trials of size nxnxn (
    3D array), of int x int x floats
     - eeg_epochs_nontarget (numpy.ndarray): Array containing epoched EEG data for
    non-target trials of size nxnxn (3D array), of int x int x floats
    """

    # our is_target_event is the same size of the number of epochs we have
    eeg_epochs_target = eeg_epochs[is_target_event]
    # same deal, index the nontargets
    eeg_epochs_nontarget = eeg_epochs[is_target_event == False]
    return eeg_epochs_target, eeg_epochs_nontarget


def plot_erps(target_erp, nontarget_erp, erp_times):
    """
    Plots the ERPs for target and non-target trials.
    This is assuming a total of 9 channels, so a 3x3 subplot is created.
    This is again, based off of Guger et al 2009.

    Parameters:
    -target_erp (numpy.ndarray): Array containing ERP data for target trials, of size n x n x n  (3D
    array), of int x int x floats
    - nontarget_erp (numpy.ndarray): Array containing ERP data for non-target trials of
    size n x n x n  (3D array), of int x int x floats
    - erp_times (numpy.ndarray): Array containing the time in
    seconds of each sample in the epoch relative to the event onset, of size n x n and floats
    """
    # find epoch_index based on which one is bigger
    # for more flexible coe

    # create our figure
    fig, axs = plt.subplots(3 ,3)
    alpha_val = 0.8
    mean_non_target = np.mean(nontarget_erp, axis=0)
    mean_target_erp = np.mean(target_erp, axis = 0)
    # iterate through our number of channels

    # Plot ERPs for non-target trials
    axs[0, 0].plot(erp_times, mean_non_target[0], color='blue', alpha = alpha_val, label='Non_target')

    axs[0, 1].plot(erp_times, mean_non_target[1], color='blue', alpha = alpha_val, label='Non_target')

    axs[0, 2].plot(erp_times, mean_non_target[2], color='blue', alpha = alpha_val, label='Non_target')

    axs[1, 0].plot(erp_times, mean_non_target[3], color='blue', alpha = alpha_val, label='Non_target')

    axs[2, 0].plot(erp_times, mean_non_target[4], color='blue', alpha = alpha_val, label='Non_target')

    axs[1, 1].plot(erp_times, mean_non_target[5], color='blue', alpha = alpha_val, label='Non_target')

    axs[2, 1].plot(erp_times, mean_non_target[6], color='blue', alpha = alpha_val, label='Non_target')

    axs[1, 2].plot(erp_times, mean_non_target[7], color='blue', alpha = alpha_val, label='Non_target')

    # Plot ERPs for target trials
    axs[0, 0].plot(erp_times, mean_target_erp[0], color='red', alpha = alpha_val, label='target')

    axs[0, 1].plot(erp_times, mean_target_erp[1], color='red', alpha = alpha_val, label='target')

    axs[0, 2].plot(erp_times, mean_target_erp[2], color='red', alpha = alpha_val, label='target')

    axs[1, 0].plot(erp_times, mean_target_erp[3], color='red', alpha = alpha_val, label='target')

    axs[2, 0].plot(erp_times, mean_target_erp[4], color='red', alpha = alpha_val, label='target')

    axs[1, 1].plot(erp_times, mean_target_erp[5], color='red', alpha = alpha_val, label='target')

    axs[2, 1].plot(erp_times, mean_target_erp[6], color='red', alpha = alpha_val, label='target')

    axs[1, 2].plot(erp_times, mean_target_erp[7], color='red', alpha = alpha_val, label='target')
    # Iterate over rows and columns of the subplot grid
    for i in range(3):
        for j in range(3):
            # Calculate the channel index based on subplot position
            channel_index = i * 3 + j

            # Set the title, xlabel, and ylabel
            axs[i, j].set_title(f'Channel {channel_index}')
            axs[i, j].set_xlabel('Time (seconds)')
            axs[i, j].set_ylabel('Voltage (uV)')

            # Add legend, grid, and vertical/horizontal lines at x=0 and y=0
            axs[i, j].legend()
            axs[i, j].grid()
            axs[i, j].axvline(x=0, color='black', linestyle='--')
            axs[i, j].axhline(y=0, color='black', linestyle='--')
    plt.tight_layout()
    fig.suptitle(' P300 Trainning ERPs ')
    





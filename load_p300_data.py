"""
Andres Segura & Lincoln Lewis
BME 6770: BCI's Lab 01
Dr. David Jangraw
2/7/2024

This module provides functions for processing and analyzing raw EEG data from the subjects in the Guger et al. 2009 paper.

The get_events function extracts event samples and corresponding target information from rowcol_id and is_target arrays.
The epoch_data function epochs the EEG data based on the event samples.
The get_erps function separates the epoched EEG data into target and non-target trials.
The plot_erps function plots the ERPs for target and non-target trials.

"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from course_software import loadmat as lm

def load_training_eeg(subject="3", data_directory="course_software/P300Data/"):
    """
    Extracts rows of raw EEG data from the given train_data.

    Parameters:
    - subject (str): Subject number (default is "3").
    - data_directory (str): Directory containing data files (default is "course_software/P300Data/").

    Returns:
    - eeg_time (numpy.ndarray): Array containing EEG time data, of size 1xn of floats, in seconds
    - eeg_data (numpy.ndarray): Array of size n x [n x 1] containing EEG data of n channels, in uV
    - rowcol_id (numpy.ndarray): Array containing row and column IDs, of integer type and size n x 1
    - is_target (numpy.ndarray): boolean array of size n x 1 recognizing whether or not a column/row id flashed has the target letter
    """
    # using string concatenation to combine the strings
    data_file = f"{data_directory}s{subject}.mat"
    # now we will extract our rows of raw data
    # load the data using loadmats built in func
    data = lm.loadmat(data_file)
    # extract the training data
    train_data = data[f"s{subject}"]["train"]

    # assigning our values
    eeg_time = np.array(train_data[0, :])
    eeg_data = np.array(train_data[1:9, :])
    rowcol_id = np.array(train_data[9, :], dtype=int)
    is_target = np.array(train_data[10, :], dtype=bool)
    # index slices is based on and in accordance with Guger et al. 2009
    return eeg_time, eeg_data, rowcol_id, is_target


def plot_raw_eeg(eeg_time, eeg_data, rowcol_id, is_target, subject_number="3"):
    """
    Plots raw eeg data for a given subject, along with the rowcolid, is_target
    all on a shared x axis

    Parameters:
    - eeg_time (numpy.ndarray): Array containing EEG time data, of size 1xn of floats, in seconds
    - eeg_data (numpy.ndarray): Array of size n x [n x 1] containing EEG data of n channels, in uV
    - rowcol_id (numpy.ndarray): Array containing row and column IDs, of integer type and size n x 1
    - is_target (numpy.ndarray): boolean array of size n x 1 recognizing whether or not a column/row id flashed has the target letter
    - subject_number (str): Subject number (default is '3'). should only be single character
    """

    # Create a new figure with 3 subplots and linked x-axes

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10), clear=True)

    # Plot rowcol_id on the first subplot
    axs[0].plot(eeg_time, rowcol_id, label="rowcol_id")
    axs[0].set_ylabel("rowcol_id")
    axs[0].legend()
    axs[0].set_title(f"Subject{subject_number}")
    axs[0].grid()


    # Plot is_target on the second subplot
    axs[1].plot(eeg_time, is_target, label="is_target")
    axs[1].set_ylabel("is_target")
    axs[1].legend()
    axs[1].grid()

    # Plot eeg_data on the third subplot

    for i in range(eeg_data.shape[0]):
        axs[2].plot(eeg_time, eeg_data[i], label=f"eeg_data_{i}")
    axs[2].set_ylabel("Voltage (uV)")
    axs[2].legend()
    axs[2].grid()
    # Set common labels and show the plot
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"plots/P300_S{subject_number}_training_rawdata.png")


def load_and_plot_all(subjects, data_directory="course_software/P300Data/"):
    """
    Loads and plots raw EEG data for multiple subjects using a simple for loop
    and our helper plotting function

    Parameters:
    - subjects (list): List of subject numbers of integer type of size n x 1
    - data_directory (str): Directory containing data files (default is "course_software/P300Data/").
    """

    # iterating through each index and subject in the list of subjects
    for subject_step in subjects:
        # using helper function to get the raw data
        eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(
            subject_step, data_directory
        )
        # and another helper function to plotg
        plot_raw_eeg(eeg_time, eeg_data, rowcol_id, is_target, subject_step)


def confirm_P300_speller(speller_matrix, is_target, rowcol_id, eeg_time, word_length=5, CPM = False):
    """
    Confirm the P300 speller word based on the provided parameters
    Finds the indexes of the is_target and indexes rowcol_id with those indexes,
    This gives us a set of 'coordinates'
    that are split into epochs based on the word_length, so that each epoch has a set of two 'coordinates'
    representing the letter that the participant was asked to spell.


    Parameters:
    - speller_matrix (list of lists): The 2D matrix representing the speller, of size nxn, each list of int type
        This matrix can be incomplete (the last index of the last row is empty) and will be populated with an empty string
    - is_target (numpy.ndarray): Boolean array indicating whether or not a target appears
    - rowcol_id (numpy.ndarray): Array of size nxn of integers representing what row/col id appears to the participant
    - word_length (int): Length of the target word to be confirmed (default is 5). This must be determined with the truth data... The amount of 'epochs'
    - CPM (booL): Default is False. If true, the function will try and calcualte the characters per minute that the participant tried to type.
        The functionality of this part is not great, as the method used to find the indexes of the epoched data relies on hardcoding a value
        based on the seperation of flashes - normally around 160ms.

    Returns:
    - str: The confirmed experimental word
    - CPM : float - the estimated characters per minute that the participant 'typed'.
    """

    # we will check to see if the matrix is full here,
    matrix_final_row = speller_matrix[len(speller_matrix) - 1]
    # checks to see if the previous row is the same size as the last one, if it is not,
    # we will say it is an incomplete matrix.
    if matrix_final_row != speller_matrix[len(speller_matrix) - 2]:
        speller_matrix[len(speller_matrix) - 1].append(" ")
        # fill the last index for our purposes

    # finding the indexes were a target appears
    target_indexes = np.where(is_target)[0]
    # using those indexes to find the rowcol_id
    target_rowcol_id = rowcol_id[target_indexes]

    # split the arrays in an even distribution based on TODO: finish
    split_rowcol_id = np.split(target_rowcol_id, word_length)

    # an empty set of coordinates
    coordinates = []
    for rowcol_array in split_rowcol_id:
        # get the counts of each character (should be identical)
        char_count = Counter(rowcol_array)
        # get the unique coordinates
        unique_coords = list(char_count.keys())
        # appends those to our empty set of coordinates
        coordinates.append(unique_coords)

    # now we have the 'coordinates' for each letter,
    # we can iterate through each coordinates list
    experimental_word = []
    # and populate our word
    for coordinate in coordinates:
        # Check if at least one element is greater than 6
        coordinate = np.sort(coordinate)  # sort first
        coordinate[1] -= 6
        # now append that letter to our expiremental word
        letter = speller_matrix[coordinate[1] - 1][coordinate[0] - 1]
        experimental_word.append(letter)
    # create the word
    experimental_word = "".join(experimental_word)
    # we will return a word that the participants were supposed to spell
    if CPM:
        # if we want to calculate CPM
        count = 0  # Count of non-zero elements\
        #initialize some values for our purposes
        n = len(rowcol_id)
        index_start_values = []
        index_end_values = []

        # make a copy of the rowcol_ids
        rowcol_id_pure = np.array(rowcol_id)

        # Traverse the array. If element
        # encountered is non-zero, then
        # replace the element at index
        # 'count' with this element
        bool = False
        for i in range(n):
            if rowcol_id_pure[i] != 0:
                # the '48' is simply the index difference of 160ms -
                # unfortunately hardcoded - ran out of time
                if rowcol_id_pure[i + 48] != 0 and bool == False:
                    bool = True
                    # append the index if we are just starting a letter
                    index_start_values.append(i)
                    # the hardcoded 48 is definitely the problem here,
                    # will only work for subject 3 - I think this depends a lot on the
                    # expirement - and is very variable.
                elif rowcol_id_pure[i + 48] == 0:
                    # append the index if we are lso just ending a letter
                    index_end_values.append(i)
                    bool = False
                # here count is incremented
                rowcol_id_pure[count] = rowcol_id_pure[i]
                count += 1

        # Now all non-zero elements have been
        # shifted to front and 'count' is set
        # as index of first 0. Make all
        # elements 0 from count to end.
        while count < n:
            rowcol_id_pure[count] = 0
            count += 1

        # setting initial values
        total_time = 0
        # This is to just get one index list of the same size as the word.
        # again, most likely because of my hardcoded 48... :(
        index_end_values_ = index_end_values[::word_length - 1]
        # the difference in indexes of 160ms should be

        # now we need to find time based off our start and stop indexes
        for index in range(len(index_start_values)):
            # setting the start and stop
            start = index_start_values[index]
            stop = index_end_values_[index]
            # total time is equal to the difference in the end and beginning of
            # epochs
            total_time += eeg_time[stop] - eeg_time[start]

        # the total characters across all epocs is equal to whatever is not zero...
        # this is what is shown
        total_characters = len(rowcol_id[rowcol_id != 0])
        # calculate words spelled per minute
        # based on total words / total time
        char_per_min = (total_characters / total_time)
    else:
        # this is for our optional....
        char_per_min = 0
    return experimental_word, char_per_min

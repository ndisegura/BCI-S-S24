## README for MNE-Python's SSVEP dataset

Written 8/26/21 by DJ.

This dataset can be used to investigate whether steady-state visual evoked potentials (SSVEPs) can be observed in EEG signals recorded on the scalp. 2 participants watched checkerboard patterns that inverted (swapped black and white) with a constant frequency of either 12.0 Hz or 15.0 Hz. Each subject experienced 10 trials that were each 20.0 s long. 32 channels of wet EEG were recorded.

To keep the data under the GitHub file size limit, we had to put them in a zipped folder. Unzip this file and place the SSVEP_S*.npz files in a folder of your choosing.

If you've installed MNE-Python, the original dataset path and description can be found in https://mne.tools/stable/overview/datasets_index.html#ssvep-dataset . However, we've converted the core information from their complex dataset objects to a simpler python dictionary. After unzipping the SsvepData.zip file, that dictionary can be loaded using the following code:

```python
import numpy as np
# Load dictionary
data = np.load('SSVEP_S1.npz')
```
Fields can then be extracted like this:
```python
# extract variables from dictionary
eeg = data['eeg']
channels = data['channels']
fs = ...
```

**Notes:**
- The fields of the dictionary are as follows:
 - eeg: the eeg data in Volts. Each row is a channel and each column is a sample.
 - channels: the name of each channel, in the same order as the eeg matrix. So ```eeg[0,:]``` is from channel ```channels[0]```.
 - fs: the sampling frequency in Hz.
 - event_samples: the sample when each event occurred. So ```eeg[:,event_samples[0]]``` is the EEG data on all channels at the moment when the first event occurred.
 - event_durations: the durations of each event in samples. So the first event ends at ```event_samples[0]+event_durations[0]```.
 - event_types: the frequency of flickering checkerboard that started flashing (either '12hz' or '15hz') for each event. ```event_types[i]``` is the type of event that occurred at sample ```event_samples[i]```.
- This is the raw, unfiltered dataset.
- The electrodes are arranged spatially on the head according to the standard 10/10 system.

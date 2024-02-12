## README for MNE-Python's AudVis dataset

Written 11/8/21 by DJ.

This dataset can be used to explore the effects of independent components analysis and removing the sources it identifies. We'll specifically use it for removal of blink artifacts from an EEG experiment. This is not a BCI dataset, but the same techniques could be applied to one.  If you've installed MNE-Python, the original dataset path and description can be found in https://mne.tools/dev/overview/datasets_index.html#sample . Here's a description from that site:

>In this experiment, checkerboard patterns were presented to the subject into the left and right visual field, interspersed by tones to the left or right ear. The interval between the stimuli was 750 ms. Occasionally a smiley face was presented at the center of the visual field. The subject was asked to press a key with the right index finger as soon as possible after the appearance of the face.

We've extracted the core information from their complex dataset objects to a simpler python dictionary (and we've further filtered the data). That dictionary can be loaded using the following code:

```python
import numpy as np
# Load dictionary
data = np.load('AudVisData.npy').item()
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
 - `eeg`: the eeg data in Volts. Each row is a channel and each column is a sample.
 - `channels`: the name of each channel, in the same order as the eeg matrix. So ```eeg[0,:]``` is from channel ```channels[0]```.
 - `fs`: the sampling frequency in Hz.
 - `event_samples`: the sample when each event occurred. So ```eeg[:,event_samples[0]]``` is the EEG data on all channels at the moment when the first event occurred.
 - `event_types`: the event code for each event. 1 = left-ear auditory tone, 2 = right-ear auditory tone, 3=left-side visual checkerboard, 4=right-side visual checkerboard, 5=smiley face, 32=button press in response to smiley face. ```event_types[i]``` is the type of event that occurred at sample ```event_samples[i]```.
 - `unmixing_matrix`: ICA source transformation. Each row is the weights used to combine across electrodes to get the activity for a single source. Multiply `unmixing_matrix` by `eeg` to transform the data into source space and get the activity of the sources estimated by ICA.
 - `mixing_matrix`: ICA components. Each row is the weights used to combine across sources to get the activity for a single electrode. Each column is the impact of a source on all electrodes. This is what's commonly plotted in ICA topographical maps. Multiply `mixing_matrix` by the source activity to transform source data into electrode space and get the activity of the electrodes when reconstructing after ICA.
- This dataset has been filtered between 1Hz and 40Hz, a fairly common band-pass range for ERP studies.
- The electrodes are arranged spatially on the head according to the standard 10/10 system.

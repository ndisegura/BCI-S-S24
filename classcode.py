# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:17:04 2024

@author: asegura
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClassCode_S24_Class_10_STARTER.py

Adapts the Kramer/Eden bootstrapping code to
more common & current Python methods.

Created on Thu Sep 30 13:16:14 2021

@author: djangraw
"""

# Import packages
import numpy as np
from matplotlib import pyplot as plt

# declare constants
ntrials = 900
sample_count = 384
fs = 500
# Generate synthetic data
t = np.arange(sample_count)/fs
signal = np.zeros(sample_count)
signal[(t>=0.3) & (t<0.5)] = 0.2
noise = np.random.randn(ntrials,sample_count)
EEGa = signal + noise

# %% Bootstrapping Steps from Kramer & Eden Ch. 2

# Draw 1000 integers with replacement from [0, 1000)
np.random.seed(343)
i = np.random.randint(0, ntrials, size=ntrials)

# Get random trials from EEGa
# Note that names ending in numbers now have o at the end
# to get them to show up in Spyder's Variable Explorer.
EEG0o = EEGa[i]  # Create the resampled EEG.

# Calculate bootstrapped ERP
ERP0o = np.mean(EEG0o,axis=0)
ERP0o = EEG0o.mean(axis=0)  # Create the resampled ERP

# Calculate actual ERP
ERPa = np.mean(EEGa,axis=0)

# Get many bootstrapped ERPs
i = np.random.randint(ntrials, size=ntrials); # Draw integers,
EEG1 = EEGa[i];                     # ... create resampled EEG,
ERP1o = EEG1.mean(0);                # ... create resampled ERP.

i = np.random.randint(ntrials, size=ntrials); # Draw integers,
EEG2 = EEGa[i];                     # ... create resampled EEG,
ERP2o = EEG2.mean(0);                # ... create resampled ERP.

i = np.random.randint(ntrials, size=ntrials); # Draw integers,
EEG3 = EEGa[i];                     # ... create resampled EEG,
ERP3o = EEG3.mean(0);                # ... create resampled ERP.


# %% STEPS 1-3: Get Bootstrapped ERPs with a loop
def bootstrapERP(EEGdata, size=None):  # Steps 1-2
    """ Calculate bootstrap ERP from data (array type)"""
    ntrials = len(EEGdata)             # Get the number of trials
    if size == None:                   # Unless the size is specified,
        size = ntrials                 # ... choose ntrials
    i = np.random.randint(ntrials, size=size)    # ... draw random trials,
    EEG0 = EEGdata[i]                  # ... create resampled EEG,
    return EEG0.mean(0)                # ... return resampled ERP.
                                       # Step 3: Repeat 3000 times 
ERP0o = [bootstrapERP(EEGa) for _ in range(3000)]
ERP0o = np.array(ERP0o)                     # ... and convert the result to an array

    
# %% STEP 4: GET CIs

# create figure
plt.figure(3,clear=True)

# copied/adapted from Kramer/Eden
ERP0o.sort(axis=0)         # Sort each column of the resampled ERP
N = len(ERP0o)             # Define the number of samples
ciL = ERP0o[int(0.025*N)]  # Determine the lower CI
ciU = ERP0o[int(0.975*N)]  # ... and the upper CI
mnA = EEGa.mean(0)        # Determine the ERP for condition A
plt.plot(t, mnA, 'k', lw=3)   # ... and plot it
plt.plot(t, ciL, 'k:')        # ... and plot the lower CI
plt.plot(t, ciU, 'k:')        # ... and the upper CI
plt.hlines(0, 0, 1, 'b')      # plot a horizontal line at 0
                          # ... and label the axes
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('Voltage (uV)')
plt.suptitle('ERP of condition A with bootstrap confidence intervals')  



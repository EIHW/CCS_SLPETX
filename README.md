# CCS_SLPETX

(1) Python Package requirement

Pytorch == 1.9.1 
numpy
os
python_speech_features


# Algorithm

(1) First, the zscoretrain300.py file is used to find the mean and variance of the training set data, based on the static, delta, and delta-deltas of the log-mel Spectrogram.

(2) Then, ExtractMel300_40.py is used to extract features and store them in the Compare2021_s300_40_fu.pkl file.

(3) Next, run the different Ablation analysis's model.py

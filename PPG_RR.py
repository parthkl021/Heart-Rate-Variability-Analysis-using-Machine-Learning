# Import necessary libraries
import scipy.io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft
from scipy import signal
import peakutils
from scipy.signal import argrelextrema
from scipy.signal import resample
from sklearn.linear_model import LinearRegression
# Load mat file
mat = scipy.io.loadmat('bidmc_data.mat')
rr_data_list = []
num_structs = mat['data']['ref'].shape[1]

# Loop through the struct and get the RR data
for i in range(num_structs):
    rr_data = mat['data']['ref'][0, i]['params'][0, 0]['rr'][0, 0]['v']
    rr_data_list.append(rr_data)

# Convert list to numpy array
rr_data_array = np.array(rr_data_list)

# Read PLETH data from the csv file
data = pd.read_csv('bidmc_csv/bidmc_01_Signals.csv')
ppg_data = data[' PLETH']

# Function to downsample a signal
def downsample(signal, current_sampling_rate, target_sampling_rate):
    downsampling_factor = current_sampling_rate // target_sampling_rate
    downsampled_signal = signal[::downsampling_factor]
    return downsampled_signal

# Downsample PLETH data
downsampled_signal = downsample(ppg_data, 125, 30)

# Create a time vector for original and downsampled signals
time_ppg_orignal = np.arange(0, len(ppg_data)/125, 1/125)
time_ppg_downsampled = np.arange(0, len(downsampled_signal)/30, 1/30)

# Filter the downsampled signal
cutoff_frequency = 0.3638  
nyquist_frequency = 30 / 2.0  
cutoff_fraction = cutoff_frequency / nyquist_frequency
b, a = signal.butter(5, cutoff_fraction, btype='high')
filtered_ppg_data = signal.filtfilt(b, a, downsampled_signal)

# Find peaks of the filtered signal
indices = peakutils.indexes(filtered_ppg_data, thres=0.02/max(filtered_ppg_data), min_dist=30/2.5)

# Identify preceding trough for each peak
comparator = np.less
trough_indices = argrelextrema(filtered_ppg_data, comparator)[0]
preceding_troughs = np.searchsorted(trough_indices, indices, side='right') - 1
preceding_troughs[preceding_troughs < 0] = 0

# Select only those troughs which precede a peak
trough_indices = trough_indices[preceding_troughs]
peak_amplitudes = filtered_ppg_data[indices]
trough_amplitudes = filtered_ppg_data[trough_indices]

print('Number of peaks detected:', len(indices))
print('Number of troughs detected:', len(trough_indices))


# Compute difference between peaks and preceding troughs
differences = peak_amplitudes - trough_amplitudes

# Calculate mean of differences for Baseline Wander (BW)
# BW was extracted as the mean amplitude between the peaks and preceding troughs;
BW_list = []
for difference in differences:
    BW_list.append(np.mean(difference))

# Calculate differences between peaks and previous troughs
amplitude_differences = []
for i in range(len(indices)):
    if indices[i] > 0:
        prev_trough_index = trough_indices[i] - 1
        difference = filtered_ppg_data[indices[i]] - filtered_ppg_data[prev_trough_index]
        amplitude_differences.append(difference)

print('Number of amplitude differences:', len(amplitude_differences))
# Calculate inter-peak intervals
peak_times = time_ppg_downsampled[indices]
inter_peak_intervals = np.diff(peak_times)

# Group RR estimates and calculate their standard deviation
rr_estimates = [inter_peak_intervals, amplitude_differences[:730], BW_list[:730]]
sd_rr = np.std(rr_estimates)

rr = rr_data_array[0][0][0]
# rr to numpy array
rr = np.array(rr)
rr = rr.flatten()

cv = np.std(amplitude_differences) / np.mean(amplitude_differences)
fn1 = cv
# the mean peak-to-peak amplitude
mean_ppa = np.mean(amplitude_differences)
fn2 = mean_ppa
# the coefficient of variation of the time between successive troughs
trough_times = time_ppg_downsampled[trough_indices]
time_diff_troughs = np.diff(trough_times)
mean_time_diff_troughs = np.mean(time_diff_troughs)
std_time_diff_troughs = np.std(time_diff_troughs)
cv_troughs = (std_time_diff_troughs / mean_time_diff_troughs)
fn3 = cv_troughs
# Identify all local maxima and minima in the filtered PPG signal
all_maxima_indices = argrelextrema(filtered_ppg_data, np.greater)[0]
all_minima_indices = argrelextrema(filtered_ppg_data, np.less)[0]
num_true_maxima = len(indices)
num_true_minima = len(trough_indices)
num_all_maxima = len(all_maxima_indices)
num_all_minima = len(all_minima_indices)
ratio_true_to_all_maxima = num_true_maxima / num_all_maxima
ratio_true_to_all_minima = num_true_minima / num_all_minima
ratio_both = (num_true_maxima + num_true_minima) / (num_all_maxima + num_all_minima)
fn4 = ratio_true_to_all_maxima

print(fn1, fn2, fn3, fn4)
f1, f2, f3, f4 = fn1, fn2, fn3, fn4
# Create a 2D array of features
X = np.array([[f1, f2, f3, f4]] * len(rr))

# Train the model
model = LinearRegression()
model.fit(X, rr)

# The model is now trained. You can check the learned coefficients.
alpha1, alpha2, alpha3, alpha4 = model.coef_
alpha5 = model.intercept_

print("alpha1:", alpha1)
print("alpha2:", alpha2)
print("alpha3:", alpha3)
print("alpha4:", alpha4)
print("alpha5:", alpha5)

y = rr.reshape(-1)
model = LinearRegression()
model.fit(X, y)
coefficients = model.coef_
alpha5 = model.intercept_
print(coefficients)
print("Intercept (α5):", alpha5)

predicted_rr = model.predict(X)
print("Predicted RR:", predicted_rr)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(rr, label='Actual')
plt.plot(predicted_rr, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Respiration Rate')
plt.title('Actual vs Predicted Respiration Rate')
plt.legend()
plt.show()



# rr_binned = bin_data(rr, binsize)
# print('Number of RR intervals:', len(rr_binned))
x = range(len(inter_peak_intervals))
plt.plot(x, inter_peak_intervals, label='Inter-Peak Intervals')
plt.plot(x, amplitude_differences[:730], label='Amplitude Differences')
plt.plot(x, BW_list[:730], label='Baseline Wander (BW)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Inter-Peak Intervals, Amplitude Differences, and Baseline Wander (BW)')
plt.legend()
plt.show()
# Print the results
print('Standard deviation of RR intervals:', sd_rr)
plt.figure(figsize=(12, 6))
plt.plot(time_ppg_downsampled[14000:15001], filtered_ppg_data[14000:15001], label='Filtered Signal')
plt.plot(time_ppg_downsampled[indices][683:], filtered_ppg_data[indices][683:], 'ro', label='Peaks') 
plt.plot(time_ppg_downsampled[trough_indices][683:], filtered_ppg_data[trough_indices][683:], 'go', label='Troughs') 
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered PPG Signal with Peaks and Troughs')
plt.legend(loc='best')
plt.show()
import pandas as pd
import scipy.io

from scipy.signal import argrelextrema, butter, filtfilt
import peakutils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def compute_features(ppg_data):

    # Function to downsample a signal
    def downsample(signal, current_sampling_rate, target_sampling_rate):
        downsampling_factor = current_sampling_rate // target_sampling_rate
        downsampled_signal = signal[::downsampling_factor]
        return downsampled_signal

    # Downsample PPG data
    downsampled_signal = downsample(ppg_data, 125, 30)

    # Filter the downsampled signal
    cutoff_frequency = 0.3638  
    nyquist_frequency = 30 / 2.0  
    cutoff_fraction = cutoff_frequency / nyquist_frequency
    b, a = butter(5, cutoff_fraction, btype='high')
    filtered_ppg_data = filtfilt(b, a, downsampled_signal)

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

    # Compute difference between peaks and preceding troughs
    differences = peak_amplitudes - trough_amplitudes

    # Calculate mean of differences for Baseline Wander (BW)
    BW_list = np.mean(differences)

    # Calculate differences between peaks and previous troughs
    amplitude_differences = []
    for i in range(len(indices)):
        if indices[i] > 0:
            prev_trough_index = trough_indices[i] - 1
            difference = filtered_ppg_data[indices[i]] - filtered_ppg_data[prev_trough_index]
            amplitude_differences.append(difference)

    # Calculate inter-peak intervals
    time_ppg_downsampled = np.arange(0, len(downsampled_signal)/30, 1/30)
    peak_times = time_ppg_downsampled[indices]
    inter_peak_intervals = np.diff(peak_times)

    # Compute the features
    fn1 = np.std(amplitude_differences) / np.mean(amplitude_differences)
    fn2 = np.mean(amplitude_differences)
    trough_times = time_ppg_downsampled[trough_indices]
    time_diff_troughs = np.diff(trough_times)
    fn3 = np.std(time_diff_troughs) / np.mean(time_diff_troughs)
    all_maxima_indices = argrelextrema(filtered_ppg_data, np.greater)[0]
    all_minima_indices = argrelextrema(filtered_ppg_data, np.less)[0]
    ratio_true_to_all_maxima = len(indices) / len(all_maxima_indices)
    fn4 = ratio_true_to_all_maxima

    return fn1, fn2, fn3, fn4

def flatten_rr_data(rr_data_list):
    rr = []
    for i in range(len(rr_data_list)):
        q = rr_data_list[i][0][0]
        q = np.array(q).flatten()
        rr.append(q)
    return np.array(rr)

def load_and_preprocess_data(start_index, end_index, exclude_rows=None):
    rr_data_list = []
    ppg_data = []

    # Ensure exclude_rows is a list to avoid issues later
    if exclude_rows is None:
        exclude_rows = []

    # Loop through the struct and read RR data and PPG data
    for i in range(start_index, end_index+1):
        if i in exclude_rows:
            continue

        # Read RR data
        rr_data = scipy.io.loadmat(f'bidmc_data.mat')['data']['ref'][0, i-1]['params'][0, 0]['rr'][0, 0]['v']
        rr_data_list.append(rr_data)
        # Read PPG data
        filename = f'bidmc_csv/bidmc_{str(i).zfill(2)}_Signals.csv'
        data = pd.read_csv(filename)
        ppg_data.append(data[' PLETH'])

    return rr_data_list, ppg_data

def train_model(rr_data_list, ppg_data):
    rr = flatten_rr_data(rr_data_list)

    features = []
    for i in range(len(ppg_data)):
        fn1, fn2, fn3, fn4 = compute_features(ppg_data[i])
        features.append([fn1, fn2, fn3, fn4])

    X = np.array(features)
    X_expanded = np.repeat(X, 481, axis=0)
    rr_flattened = rr.flatten()

    model = LinearRegression()
    model.fit(X_expanded, rr_flattened)

    return model, X_expanded, rr_flattened

def evaluate_model(model, X, rr_flattened):
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)
    error = np.mean(np.abs(predictions - rr_flattened))
    print("error: ", error)
    plt.plot(rr_flattened, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()

def split_data(rr_data_list, ppg_data, split):
    rr_train, rr_test, ppg_train, ppg_test = train_test_split(rr_data_list, ppg_data, test_size=split)
    return rr_train, rr_test, ppg_train, ppg_test

def train_and_evaluate_for_group(patient_group, excluded_group,split):

    # Load and preprocess data for the patient group
    rr_data, ppg_data = load_and_preprocess_data(min(patient_group), max(patient_group), exclude_rows=excluded_group)

    rr_train, rr_test, ppg_train, ppg_test = split_data(rr_data, ppg_data, split)

    model, X_train, rr_train_flattened = train_model(rr_train, ppg_train)
    print("Training evaluation for the patient group:")
    evaluate_model(model, X_train, rr_train_flattened)

    X_test = []  
    for ppg in ppg_test:
        features = compute_features(ppg)
        X_test.append(features)

    X_test_expanded = np.repeat(X_test, 481, axis=0)
    
    print("Test evaluation for the patient group:")
    rr_test_flattened = flatten_rr_data(rr_test).flatten()
    evaluate_model(model, X_test_expanded, rr_test_flattened)

    return model

age_list = []
for i in range(1, 54):
    filename = f'bidmc_csv/bidmc_{str(i).zfill(2)}_Fix.txt'
    with open(filename, 'r') as file:
        file_contents = file.read()
    age_index = file_contents.find("Age:")
    if age_index != -1:
        age_string = file_contents[age_index + len("Age:"):].strip()
        age_end_index = age_string.find("\n")
        if age_end_index != -1:
            age = age_string[:age_end_index].strip()
            age_list.append(age)
ex = []
for i in range(len(age_list)):
    if(age_list[i] == "90+"):
        age_list[i] = '91'
    if(age_list[i] == "NaN"):
        age_list[i] = '0'
        ex.append(i+1)
    age_list[i] = int(age_list[i])

less_tahn_60 = []
for i in range(len(age_list)):
    if(age_list[i] <= 60):
        less_tahn_60.append(i+1)

greater_than_60 = []
for i in range(len(age_list)):
    if(age_list[i] > 60):
        greater_than_60.append(i+1)

print("less than 60: ", len(less_tahn_60) -2)
print("greater than 60: ", len(greater_than_60) - 2)
# print(less_tahn_60)
less_than_60_model = train_and_evaluate_for_group(less_tahn_60, ex + greater_than_60 +[15] , 0.1)
greater_than_60_model = train_and_evaluate_for_group(greater_than_60, ex + less_tahn_60 + [13] + [19], 0.2)





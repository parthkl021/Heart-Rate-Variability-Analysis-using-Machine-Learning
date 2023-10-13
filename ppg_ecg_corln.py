import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
from scipy.signal import welch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.fftpack import dct, idct

data = pd.read_excel('Sensor_Data.xlsx')
row_to_exclude = [i for i in range(0, 98)] + [i for i in range(104, 116)]
data = data.drop(row_to_exclude)
# cbt_val = data['CBT(degC)']
# cbt_val = cbt_val.to_numpy()

data['ECG'] = data['ECG'].astype(str)
ecg_readings = []

for i in data['ECG']:
    data_list = i.split(':')
    ecg_readings.append([int(item) for item in data_list if item])
mean_ecg = np.mean(ecg_readings)

data['PPG'] = data['PPG'].astype(str)
ppg_readings = []
print(len(data['PPG']))
for i in data['PPG']:
    data_list = i.split(':')
    ppg_readings.append([int(item) for item in data_list if item])
mean_ppg = np.mean(ppg_readings)

time_ppg = np.linspace(0, 19, len(ppg_readings[0]))
time_ecg = np.linspace(0, 19, len(ecg_readings[0]))

print(f"Mean ECG: {mean_ecg}, Mean cbt: {mean_ppg}")

# Compute autocorrelation for each row of ECG readings
ecg_autocorr = [pd.Series(row).autocorr() for row in ecg_readings]
ppg_autocorr = [pd.Series(row).autocorr() for row in ppg_readings]

print(f"ECG Autocorrelation: {ecg_autocorr}")
print(f"PPG Autocorrelation: {ppg_autocorr}")

# Flatten your ECG readings
# Flatten your ECG readings
ecg_readings = np.ravel(ecg_readings)
sample_rate_ecg = 20
# Create a time sequence that is 20/200 seconds apart for ECG
total_time_duration_ecg = len(ecg_readings) * (sample_rate_ecg/200)
time_array_ecg = np.linspace(0, total_time_duration_ecg, len(ecg_readings))

# Flatten your PPG readings
ppg_readings = np.ravel(ppg_readings)
sample_rate_ppg = 20
# Create a time sequence that is 20/100 seconds apart for PPG
total_time_duration_ppg = len(ppg_readings) * (sample_rate_ppg/100) 
time_array_ppg = np.linspace(0, total_time_duration_ppg, len(ppg_readings))

ppg_series = pd.Series(ppg_readings, index=pd.date_range(start='2023-01-01', periods=len(ppg_readings), freq='D'))
ecg_series = pd.Series(ecg_readings, index=pd.date_range(start='2023-01-01', periods=len(ecg_readings), freq='D'))

assert len(ecg_readings) % len(ppg_readings) == 0
ratio = len(ecg_readings) // len(ppg_readings)
ecg_readings_reshaped = np.reshape(ecg_readings, (-1, ratio))
ecg_readings_aggregated = np.mean(ecg_readings_reshaped, axis=1)
assert len(ecg_readings_aggregated) == len(ppg_readings)

# Min-Max normalization
ecg_readings_normalized = (ecg_readings_aggregated - np.min(ecg_readings_aggregated)) / (np.max(ecg_readings_aggregated) - np.min(ecg_readings_aggregated))
ppg_readings_normalized = (ppg_readings - np.min(ppg_readings)) / (np.max(ppg_readings) - np.min(ppg_readings))

# corelation
correlation_matrix = np.corrcoef(ecg_readings_normalized, ppg_readings_normalized)
correlation_coefficient = correlation_matrix[0, 1]

print(f"The correlation coefficient between the normalizrd ECG and PPG readings is {correlation_coefficient}")

fig, ax = plt.subplots()
ax.plot(time_array_ppg, ecg_readings_normalized, label='ECG')
ax.plot(time_array_ppg, ppg_readings_normalized, label='PPG')
plt.xlabel('Time (seconds)')
plt.ylabel('Signal')
plt.title('ECG_mod and ECG Signal over Time')
plt.legend()
plt.show()

# Compute the DCT of each cycle of ECG and PPG readings
ecg_dct = dct(ecg_readings_normalized)
ppg_dct = dct(ppg_readings_normalized)

# Train a linear regression model that maps PPG DCT coefficients to ECG DCT coefficients
reg = LinearRegression().fit(ppg_dct.reshape(-1, 1), ecg_dct)
# Predict the ECG DCT coefficients from the PPG DCT coefficients
ecg_dct_pred = reg.predict(ppg_dct.reshape(-1, 1))
# Obtain the predicted ECG readings via the inverse DCT
ecg_pred = idct(ecg_dct_pred)
plt.figure(figsize=(12, 6))
plt.plot(time_array_ppg, ecg_readings_aggregated, label='Actual aggregated ECG')
plt.plot(time_array_ppg, ecg_pred, label='Predicted ECG')
plt.xlabel('Time (s)')
plt.ylabel('ECG Reading')
plt.legend()
plt.show()

# Power spectral analysis
freq_ppg, psd_ppg = welch(ppg_dct, nperseg=len(ppg_dct))
freq_ecg, ecg_cbt = welch(ecg_dct, nperseg=len(ecg_dct))
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(freq_ppg, psd_ppg)
plt.xlabel('Frequency')
plt.ylabel('Power Spectral Density')
plt.title('PPG Power Spectral Density')
plt.subplot(2, 1, 2)
plt.plot(freq_ecg, ecg_cbt)
plt.xlabel('Frequency')
plt.ylabel('Power Spectral Density')
plt.title('ECG Power Spectral Density')
plt.show()

# Trend analysis
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(ppg_dct, marker='o')
plt.xlabel('Time')
plt.ylabel('PPG')
plt.title('PPG Trend')

plt.subplot(2, 1, 2)
plt.plot(ecg_dct, marker='o')
plt.xlabel('Time')
plt.ylabel('ECG')
plt.title('ECG Trend')
plt.show()

ppg_series = pd.Series(ppg_dct, index=pd.date_range(start='2023-01-01', periods=len(ppg_readings), freq='D'))
ecg_series = pd.Series(ecg_dct, index=pd.date_range(start='2023-01-01', periods=len(ecg_readings_aggregated), freq='D'))

# Seasonal decomposition
decomposition = seasonal_decompose(ppg_series, model='additive')
seasonal = decomposition.seasonal
trend = decomposition.trend
residual = decomposition.resid

# Decomposition for ecg_series
decomposition_ecg = seasonal_decompose(ecg_series, model='additive')
seasonal_ecg = decomposition_ecg.seasonal
trend_ecg = decomposition_ecg.trend
residual_ecg = decomposition_ecg.resid

# Plotting PPG and ECG together
plt.figure(figsize=(14, 8))

# PPG original
plt.subplot(4, 2, 1)
plt.plot(ppg_series, marker='o', label='PPG')
plt.xlabel('Time')
plt.ylabel('PPG')
plt.title('PPG')

# ECG original
plt.subplot(4, 2, 2)
plt.plot(ecg_series, marker='o', color='r', label='ECG')
plt.xlabel('Time')
plt.ylabel('ECG')
plt.title('ECG')

# PPG trend
plt.subplot(4, 2, 3)
plt.plot(trend, marker='o', label='PPG Trend')
plt.xlabel('Time')
plt.ylabel('Trend')
plt.title('PPG Trend')

# ECG trend
plt.subplot(4, 2, 4)
plt.plot(trend_ecg, marker='o', color='r', label='ECG Trend')
plt.xlabel('Time')
plt.ylabel('Trend')
plt.title('ECG Trend')

# PPG seasonal
plt.subplot(4, 2, 5)
plt.plot(seasonal, marker='o', label='PPG Seasonal')
plt.xlabel('Time')
plt.ylabel('Seasonal')
plt.title('PPG Seasonal')

# ECG seasonal
plt.subplot(4, 2, 6)
plt.plot(seasonal_ecg, marker='o', color='r', label='ECG Seasonal')
plt.xlabel('Time')
plt.ylabel('Seasonal')
plt.title('ECG Seasonal')

# PPG residual
plt.subplot(4, 2, 7)
plt.plot(residual, marker='o', label='PPG Residual')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('PPG Residual')

# ECG residual
plt.subplot(4, 2, 8)
plt.plot(residual_ecg, marker='o', color='r', label='ECG Residual')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('ECG Residual')

# Adjust spacing between plots
plt.tight_layout()
plt.show()

# We first preprocess the ECG and PPG signal pairs to obtain temporally aligned 
# and normalized sets of signals. We then segment the signals into pairs of cycles 
# and train a linear transform that maps the discrete cosine transform (DCT) coefficients 
# of the PPG cycle to those of the corresponding ECG cycle. The ECG waveform is then
# obtained via the inverse DCT.
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

input_path = Path("../recordings/LP_Lick10_FN.wav")
output_dir = Path("../Exp1/Audio_outputs")
plot_dir = Path("../Exp1/plots")

x, fs = sf.read(str(input_path))

if x.ndim == 2:
    x = np.mean(x, axis=1)

N = len(x)
duration = N / fs
time = np.arange(N) / fs

print("Sampling frequency:", fs, "Hz")
print("Number of samples:", N)
print("Duration:", duration, "seconds")

plt.figure(figsize=(12, 4))
plt.plot(time, x)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Original Guitar Audio")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir/"Original Guitar Audio.png", dpi = 300)
plt.show()

# ==========================================
# PART 2: CREATE ROOM IMPULSE RESPONSE
# ==========================================

def create_impulse_response(amplitudes, delays_seconds, fs):
    if len(amplitudes) != len(delays_seconds):
        raise ValueError("Amplitudes and delays must have equal lengths.")

    if any(a < 0 for a in amplitudes):
        raise ValueError("Amplitudes must be nonnegative.")

    if any(d < 0 for d in delays_seconds):
        raise ValueError("Delays must be nonnegative.")

    # Convert delays from seconds to samples
    delay_samples = []

    for delay in delays_seconds:
        d = int(round(delay * fs))
        delay_samples.append(d)

    # Create the impulse response
    h_length = max(delay_samples, default=0) + 1
    h = np.zeros(h_length)
    h[0] = 1.0

    # Add reflections
    for i in range(len(amplitudes)):
        h[delay_samples[i]] += amplitudes[i]

    return h, delay_samples


# Experimental parameters
scenario = "B"
amplitudes = [0.5, 0.4, 0.3, 0.2, 0.1]
delays_seconds = [0.03, 0.055, 0.08, 0.110, 0.145]

h, delay_samples = create_impulse_response(
    amplitudes, delays_seconds, fs
)

print("Reflection delays in samples:", delay_samples)
print("Impulse response length:", len(h))

# Plot impulse response
impulse_time = np.arange(len(h)) / fs
plt.figure(figsize=(12, 4))
plt.stem(impulse_time, h)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Room Impulse Response")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir/f"{scenario}_Impulse_response.png", dpi = 300)
plt.show()

# ==========================================
# PART 3: MANUAL CONVOLUTION
# ==========================================

def manual_convolution(x, h):
    N = len(x)
    M = len(h)

    y = np.zeros(N + M - 1)

    # Add each scaled, delayed copy of the input
    for k in range(M):
        y[k:k + N] += h[k] * x

    return y


print("Applying convolution...")

y = manual_convolution(x, h)

print("Convolution completed!")
print("Original samples:", len(x))
print("Output samples:", len(y))
print("Output duration:", len(y) / fs, "seconds")

# ==========================================
# PART 4: PLOT AND SAVE WAVEFORMS
# ==========================================

time_original = np.arange(len(x)) / fs
time_reverb = np.arange(len(y)) / fs

# Original waveform
fig2 = plt.figure(figsize=(14, 5))
plt.subplot(2,1,1)
plt.plot(time_original, x)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Original Guitar Waveform")
plt.grid(True)
plt.tight_layout()


# Reverberated waveform
plt.subplot(2,1,2)
plt.plot(time_reverb, y)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Reverberated Guitar Waveform")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir / f"{scenario}_waveform_comparison.png", dpi=300)
plt.show()
plt.close(fig2)

print("Both waveform plots saved successfully!")

# ==========================================
# PART 6: SAVE REVERBERATED AUDIO
# ==========================================

# Prevent clipping when saving the audio
peak = np.max(np.abs(y))

if peak > 1.0:
    y_to_save = y / peak
else:
    y_to_save = y

sf.write(str(output_dir / f"{scenario}_reverberated_guitar.wav"), y_to_save, fs)

print("Reverberated audio saved as reverberated_guitar.wav")

import math

# ==========================================
# MANUAL DFT IMPLEMENTATION
# ==========================================

def dft(x):
    N = len(x)
    X = []

    for k in range(N):
        real = 0.0
        imag = 0.0

        for n in range(N):
            angle = -2.0 * math.pi * k * n / N

            real += x[n] * math.cos(angle)
            imag += x[n] * math.sin(angle)

        X.append(complex(real, imag))

    return X

# ==========================================
# CALCULATE MAGNITUDE SPECTRUM USING DFT
# ==========================================

def dft_magnitude(x, fs):
    N = len(x)

    if N < 2:
        return [], []

    # Compute DFT
    X = dft(x)

    frequencies = []
    magnitudes = []

    # Only positive frequencies
    for k in range(N // 2 + 1):

        frequency = k * fs / N
        magnitude = abs(X[k]) / N

        # Single-sided amplitude spectrum
        if k != 0 and not (N % 2 == 0 and k == N // 2):
            magnitude *= 2

        frequencies.append(frequency)
        magnitudes.append(magnitude)

    return frequencies, magnitudes

threshold = 0.001
active_indices = np.where(np.abs(x) > threshold)[0]

if len(active_indices) == 0:
    raise ValueError("No active guitar samples found.")

start_index = int(active_indices[0]) 

print("Guitar starts at:", start_index / fs, "seconds")

segment_length = 2048

x_segment = x[start_index:start_index + segment_length]
y_segment = y[start_index:start_index + segment_length]

if len(x_segment) < segment_length or len(y_segment) < segment_length:
    raise ValueError("Not enough samples available for DFT analysis.")

print("DFT segment samples:", len(x_segment))
print("DFT segment duration:", len(x_segment) / fs, "seconds")

print("Calculating original magnitude spectrum using DFT...")

freq_x, mag_x = dft_magnitude(
    x_segment, fs
)

print("Calculating reverberated magnitude spectrum using DFT...")

freq_y, mag_y = dft_magnitude(
    y_segment, fs
)

print("Both DFT magnitude spectra calculated!")

# ==========================================
# GUITAR DFT SPECTRUM
# ==========================================

plt.figure(figsize=(12, 5))
plt.plot(freq_x, mag_x,label = "Magnitude Spectrum - Original Guitar")
plt.plot(freq_y, mag_y,label = "Magnitude Spectrum - Reverberated Guitar")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir / f"{scenario}_spectra_comparison.png", dpi=300)
plt.show()
plt.close()
print("DFT magnitude spectrum plots saved successfully!")
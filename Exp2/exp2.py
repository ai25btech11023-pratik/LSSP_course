# Experiment 2: Musical effects using LTV Systems

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# file paths
input_path = "../recordings/LP_Lick10_FN.wav"
output_dir = Path("../Exp2/Audio_outputs")
plot_dir = Path("../Exp2/plots")
output_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

x, fs = sf.read(input_path)

# average the channels if the recording is stereo
if x.ndim == 2:
    x = np.mean(x, axis=1)

N = len(x)
duration = N / fs
time = np.arange(N) / fs

print("Sampling frequency:", fs, "Hz")
print("Number of samples:", N)
print("Duration:", duration, "seconds")
print("Input peak amplitude:", np.max(np.abs(x)))

# plot the original recording
plt.figure(figsize=(12, 4))
plt.plot(time, x)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Original Guitar Audio")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir /"original_guitar_waveform.png", dpi=300)
plt.show()
plt.close()

# time-varying delay

def time_varying_delay(x, fs, alpha=0.5, D0_ms=15.0,
                       deltaD_ms=5.0, fm=0.5):
    
    # estimate delayed samples using linear interpolation

    x = np.asarray(x, dtype=float)
    n_samples = len(x)
    n = np.arange(n_samples)

    D0 = D0_ms * fs / 1000.0
    deltaD = deltaD_ms * fs / 1000.0
    D = D0 + deltaD * np.sin(2.0 * np.pi * fm * n / fs)

    y = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        source_position = i - D[i]

        if source_position < 0:
            delayed_sample = 0.0
        else:
            left = int(math.floor(source_position))
            right = left + 1
            fraction = source_position - left

            if right >= n_samples:
                delayed_sample = x[left]
            else:
                delayed_sample = (1.0 - fraction) * x[left] + fraction * x[right]

        y[i] = x[i] + alpha * delayed_sample

    return y, D


# Main experimental parameters
alpha = 0.5
D0_ms = 15.0
deltaD_ms = 5.0
fm = 0.5

y, D_n = time_varying_delay(
    x, fs, alpha=alpha, D0_ms=D0_ms,
    deltaD_ms=deltaD_ms, fm=fm
)

print("Time-varying delay processing completed.")
print("Output samples:", len(y))
print("Output duration:", len(y) / fs, "seconds")
print("Output peak amplitude", np.max(np.abs(y)))

# Plot the delay in milliseconds
delay_time = np.arange(len(D_n)) / fs
plot_count = min(len(D_n), int(3 * fs))

plt.figure(figsize=(12, 4))
plt.plot(delay_time[:plot_count], D_n[:plot_count] * 1000.0 / fs)
plt.xlabel("Time (seconds)")
plt.ylabel("Delay (milliseconds)")
plt.title("Time-Varying Delay")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir /"delay_curve.png", dpi=300)
plt.show()
plt.close()


# manual dft

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

# calculate the magnitude spectrum

def dft_magnitude(x, fs):
    N = len(x)

    if N < 2:
        return [], []

    X = dft(x)

    frequencies = []
    magnitudes = []

    # keep zero and positive frequencies
    for k in range(N // 2 + 1):

        frequency = k * fs / N
        magnitude = abs(X[k]) / N

        # single-sided magnitude
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

# save the processed audio

# reduce the saved audio level if needed
peak = float(np.max(np.abs(y))) if len(y) else 0.0
if peak > 1.0:
    y_save = y / peak
    print("Applied global output gain: ", 1.0 / peak)
else:
    y_save = y

output_audio_path = output_dir / f"processed_guitar(fm{fm}_dD{deltaD_ms}).wav"
sf.write(str(output_audio_path), y_save, fs, subtype="PCM_16")


# output with this modulation frequency

fm = 0.5  # Modulation frequency in Hz

# Apply time-varying delay with fm Hz

y_test, _ = time_varying_delay(
    x, fs, alpha=alpha, D0_ms=D0_ms,
    deltaD_ms=deltaD_ms, fm=fm
)
segment_length = 2048
y_segment = y_test[start_index:start_index + segment_length]

print("Calculating processed spectrum...")
freq_y, mag_y = dft_magnitude(y_segment, fs)
print("spectra calculated.")

# compare the original and processed waveforms

N = len(x)
t = np.arange(N) / fs

plt.figure(figsize=(12, 5))
plt.subplot(2,1,1)
plt.plot(t, x[:N])
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("original waveform")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.subplot(2,1,2)
plt.plot(t, y_test[:N],)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title(f"time delay effect at fm = {fm}Hz")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir /f"waveform_comparison_fm{fm}.png", dpi=300)
plt.show()
plt.close()


# compare the magnitude spectra
plt.figure(figsize=(12, 5))
plt.plot(freq_x, mag_x, label="Original guitar")
plt.plot(freq_y, mag_y, label="Time-varying delay output")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Single-sided magnitude")
plt.title(f"magnitude_spectra_comparison_fm{fm} Hz")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir /f"spectra_comparison_fm{fm}.png", dpi=300)
plt.show()
plt.close()

# try a smaller delay variation

deltaD_ms = 2.0

# Apply time-varying delay with delay variation = deltaD_ms
y_test, _ = time_varying_delay(
    x, fs, alpha=alpha, D0_ms=D0_ms,
    deltaD_ms=deltaD_ms, fm=fm
)

segment_length = 2048

y_segment = y_test[start_index:start_index + segment_length]

print("Calculating processed spectrum...")
freq_y, mag_y = dft_magnitude(y_segment, fs)
print("spectra calculated.")

# compare the original and processed waveforms
N = len(y_test)
t = np.arange(N) / fs

fig2 = plt.figure(figsize=(12, 5))
plt.subplot(2,1,1)
plt.plot(t, x[:N])
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("original waveform")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(t, y_test[:N])
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title(f"waveform_delay_var{deltaD_ms} ms")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(plot_dir / f"waveform_comparison_dD{deltaD_ms}.png", dpi=300)
plt.show()
plt.close(fig2)

# compare the magnitude spectra

plt.figure(figsize=(12, 5))
plt.plot(freq_x, mag_x, label="Original guitar")
plt.plot(freq_y, mag_y, label="Time-varying delay output")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Single-sided magnitude")
plt.title(f"magnitude_spectra_comparison_dD{deltaD_ms} ms")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir /f"spectra_comparison_dD{deltaD_ms}.png", dpi=300)
plt.show()
plt.close()




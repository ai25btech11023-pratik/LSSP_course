# Experiment 2: Musical effects using LTV Systems

# y[n] = x[n] + \delta_D x[n-D[n]], \qquad D[n] = D_0 + \Delta D\sin(2\pi f_m n/f_s). 

# ====================================================
# IMPORTS and PREPROCESSING THE GUITAR AUDIO

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# I/O paths
input_path = "../recordings/LP_Lick10_FN.wav"
output_dir = Path("../Exp2/Audio_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

x, fs = sf.read(input_path)

# Convert stereo audio to mono by averaging the channels.
if x.ndim == 2:
    x = np.mean(x, axis=1)

N = len(x)
duration = N / fs
time = np.arange(N) / fs

print("Sampling frequency:", fs, "Hz")
print("Number of samples:", N)
print("Duration:", duration, "seconds")
print("Input peak amplitude:", np.max(np.abs(x)))

plt.figure(figsize=(12, 4))
plt.plot(time, x)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Original Guitar Audio")
plt.grid(True)
plt.tight_layout()
plt.savefig("../Exp2/plots/original_guitar_waveform.png", dpi=300)
plt.show()
plt.close()

# ====================================================
# IMPLEMENT THE TIME-VARYING DELAY SYSTEM

def time_varying_delay(x, fs, alpha=0.5, D0_ms=15.0,
                       deltaD_ms=5.0, fm=0.5):
    
    # Apply y[n] = x[n] + delta_D * x[n-D[n]] with linear interpolation.

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
plt.savefig("../Exp2/plots/delay_curve.png", dpi=300)
plt.show()
plt.close()


# ====================================================
# MANUAL FFT IMPLEMENTATION
# Radix-2 Cooley-Tukey algorithm

samples = len(x)

def fft(signal):
    """Compute the forward FFT """
    N = len(signal)
    if N == 0:
        return []
    if N & (N - 1):
        raise ValueError("FFT input length must be a power of two.")

    X = [complex(value, 0.0) for value in signal]

    # Bit-reversal permutation
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            X[i], X[j] = X[j], X[i]

    # Butterfly stages
    length = 2
    while length <= N:
        half = length // 2
        angle = -2.0 * math.pi / length
        w_real = math.cos(angle)
        w_imag = math.sin(angle)

        for start in range(0, N, length):
            wr, wi = 1.0, 0.0
            for offset in range(half):
                even_index = start + offset
                odd_index = even_index + half
                even = X[even_index]
                odd = X[odd_index]

                t = complex(
                    wr * odd.real - wi * odd.imag,
                    wr * odd.imag + wi * odd.real
                )
                X[even_index] = even + t
                X[odd_index] = even - t

                next_wr = wr * w_real - wi * w_imag
                next_wi = wr * w_imag + wi * w_real
                wr, wi = next_wr, next_wi

        length *= 2

    return X


def fft_magnitude(signal, fs, N= samples):

    N = min(N, len(signal))
    if N < 2:
        return [], []

    # Use the largest power of two no greater than the available length.
    N = 1 << (N.bit_length() - 1)
    segment = [float(signal[i]) for i in range(N)]

    # Remove DC offset before calculating the spectrum.
    mean_value = sum(segment) / N
    segment = [sample - mean_value for sample in segment]
    spectrum = fft(segment)

    frequencies = []
    magnitudes = []
    for k in range(N // 2 + 1):
        frequencies.append(k * fs / N)
        magnitude = abs(spectrum[k]) / N
        if k != 0 and k != N // 2:
            magnitude *= 2.0
        magnitudes.append(magnitude)

    return frequencies, magnitudes

print("Calculating original spectrum...")
freq_x, mag_x = fft_magnitude(x, fs)
print("Original spectra calculated.")

# ====================================================
# SAVE THE PROCESSED AUDIO

# apply one global gain
peak = float(np.max(np.abs(y))) if len(y) else 0.0
if peak > 1.0:
    y_save = y / peak
    print("Applied global output gain: ", 1.0 / peak)
else:
    y_save = y

output_audio_path = output_dir / f"processed_guitar(fm{fm}_dD{deltaD_ms}).wav"
sf.write(str(output_audio_path), y_save, fs, subtype="PCM_16")


# ====================================================
# EXPERIMENT WITH MODULATION FREQUENCY

fm = 0.5  # Modulation frequency in Hz

# Apply time-varying delay with fm Hz

y_test, _ = time_varying_delay(
    x, fs, alpha=alpha, D0_ms=D0_ms,
    deltaD_ms=deltaD_ms, fm=fm
)

print("Calculating processed spectrum...")
freq_y, mag_y = fft_magnitude(y_test, fs)
print("spectra calculated.")

# Plot the original and processed signals
N = len(x)
t = np.arange(N) / fs

plt.figure(figsize=(12, 5))
plt.plot(t, x[:N], label="Original guitar")
plt.plot(t, y_test[:N], label=f"Output (f_m = {fm} Hz)")

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title(f"waveform_comparison_fm{fm} Hz")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(f"../Exp2/plots/waveform_comparison_fm{fm}.png", dpi=300)
plt.show()
plt.close()


# Plot both spectra
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
plt.savefig(f"../Exp2/plots/spectra_comparison_fm{fm}.png", dpi=300)
plt.show()
plt.close()

# ====================================================
# EXPERIMENT WITH DELAY VARIATION

deltaD_ms = 2.0

# Apply time-varying delay with delay variation = deltaD_ms
y_test, _ = time_varying_delay(
    x, fs, alpha=alpha, D0_ms=D0_ms,
    deltaD_ms=deltaD_ms, fm=fm
)

print("Calculating processed spectrum...")
freq_y, mag_y = fft_magnitude(y_test, fs)
print("spectra calculated.")

# Plot original and processed signals
N = len(y_test)
t = np.arange(N) / fs

plt.figure(figsize=(12, 5))
plt.plot(t, x[:N], label="Original guitar")
plt.plot(t, y_test[:N], label=f"Output at delay variation {deltaD_ms} ms")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title(f"Effect of Delay Variation ({deltaD_ms} ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../Exp2/plots/waveform_comparison_dD{deltaD_ms}.png", dpi=300)
plt.show()
plt.close()

# Plot both spectra
plt.figure(figsize=(12, 5))
plt.plot(freq_x, mag_x, label="Original guitar")
plt.plot(freq_y, mag_y, label="Time-varying delay output")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Single-sided magnitude")
plt.title(f"Effect of Delay Variation ({deltaD_ms} ms)")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../Exp2/plots/spectra_comparison_dD{deltaD_ms}.png", dpi=300)
plt.show()
plt.close()



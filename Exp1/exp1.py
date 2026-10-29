# Experiment 1: Room Reverberation using an LTI System

# IMPORTS and PREPROCESSING THE GUITAR AUDIO
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# I/O paths
input_path = "../recordings/LP_Lick10_FN.wav"
output_dir = Path("../Exp1/Audio_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

x, fs = sf.read(input_path)

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
plt.savefig("../Exp1/plots/original_guitar_waveform.png", dpi=300)
plt.show()


# ==========================================
# CREATE SIMPLE IMPULSE RESPONSE
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
scenario = 1
amplitudes = [0.60, 0.35, 0.20]
delays_seconds = [0.10, 0.20, 0.30]

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
plt.title("Simple Impulse Response")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../Exp1/plots/Impulse_response{scenario}.png", dpi=300)
plt.show()


# ==========================================
# MANUAL CONVOLUTION
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
# PLOT WAVEFORMS
# ==========================================

time_original = np.arange(len(x)) / fs
time_reverb = np.arange(len(y)) / fs

# Original waveform
plt.figure(figsize=(14, 5))
plt.plot(time_original, x, label = "Original Guitar")
plt.plot(time_reverb, y, label = "Reverberated Guitar")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Waveform_comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../Exp1/plots/Waveform_comparison{scenario}.png", dpi=300)
plt.show()
plt.close()



# ==========================================
# SAVE REVERBERATED AUDIO
# ==========================================

peak = np.max(np.abs(y))

if peak > 1.0:
    y_save = y / peak
else:
    y_save = y
output_audio_path = output_dir / f"reverberated_audio{scenario}.wav"
sf.write(str(output_audio_path), y_save, fs)



import math

# ==========================================
# PART 1: MANUAL FFT IMPLEMENTATION
# Radix-2 Cooley-Tukey algorithm
# ==========================================
samples = len(y)

def fft(x):
    N = len(x)

    if N == 0:
        return []

    # FFT requires a power-of-two length
    if N & (N - 1):
        raise ValueError("FFT input length must be a power of two.")

    # Convert samples to complex numbers
    X = [complex(value, 0.0) for value in x]

    # Step 1: Bit-reversal permutation
    j = 0

    for i in range(1, N):
        bit = N >> 1

        while j & bit:
            j ^= bit
            bit >>= 1

        j ^= bit

        if i < j:
            X[i], X[j] = X[j], X[i]

    # Step 2: Iterative Cooley-Tukey butterfly stages
    length = 2

    while length <= N:
        half = length // 2

        angle = -2.0 * math.pi / length
        w_real = math.cos(angle)
        w_imag = math.sin(angle)

        for start in range(0, N, length):
            wr = 1.0
            wi = 0.0

            for offset in range(half):
                even_index = start + offset
                odd_index = even_index + half

                even = X[even_index]
                odd = X[odd_index]

                # Multiply odd branch by the twiddle factor
                t = complex(
                    wr * odd.real - wi * odd.imag,
                    wr * odd.imag + wi * odd.real
                )

                # Butterfly outputs
                X[even_index] = even + t
                X[odd_index] = even - t

                # Update twiddle factor
                next_wr = wr * w_real - wi * w_imag
                next_wi = wr * w_imag + wi * w_real

                wr = next_wr
                wi = next_wi

        length *= 2

    return X


def fft_magnitude(x, fs, N=samples):
    N = min(N, len(x))

    if N < 2:
        return [], []

    # Largest power of two <= available sample count
    N = 1 << (N.bit_length() - 1)

    segment = [float(x[n]) for n in range(N)]

    # Remove mean to reduce the DC component
    mean = sum(segment) / N
    segment = [sample - mean for sample in segment]

    # Compute FFT using our manual implementation
    X = fft(segment)

    frequencies = []
    magnitudes = []

    # Positive-frequency components
    for k in range(N // 2 + 1):
        frequency = k * fs / N
        magnitude = abs(X[k]) / N

        # Single-sided amplitude spectrum
        if k != 0 and not (N % 2 == 0 and k == N // 2):
            magnitude *= 2

        frequencies.append(frequency)
        magnitudes.append(magnitude)

    return frequencies, magnitudes



# ==========================================
# COMPUTE BOTH SPECTRA
# ==========================================

print("Calculating original magnitude spectrum...")

freq_x, mag_x = fft_magnitude(x, fs)

print("Calculating reverberated magnitude spectrum...")

freq_y, mag_y = fft_magnitude(y, fs)

print("Both magnitude spectra calculated!")



# ==========================================
# PLOT MAGNITUDE SPECTRA
# ==========================================

# Graph: magnitude spectrum
plt.figure(figsize=(12, 5))
plt.plot(freq_x, mag_x, color="blue", label = "Original Guitar")
plt.plot(freq_y, mag_y, color="orange",label = "Reverberated Guitar")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum")
plt.legend()
plt.xlim(0, 5000)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../Exp1/plots/spectra_comparison{scenario}.png", dpi=300)
plt.show()
plt.close()



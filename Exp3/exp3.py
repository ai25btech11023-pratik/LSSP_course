# Experiment 3: Two-Mode Audio Equalizer

# load audio

# Equalizer model:
#   x_L[n] = LPF{x[n]}
#   x_H[n] = x[n] - x_L[n]
#   y[n]   = gL*x_L[n] + gH*x_H[n]
#
# Modes:
#   1. Original
#   2. Drum Enhance
#   3. Guitar Enhance
#   4. Drum + Guitar Enhance

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# ====================================================
# FILE PATHS
# ====================================================

input_path = Path("../recordings/rock.00028.wav")
output_dir = Path("../Exp3/Audio_outputs")
plot_dir = Path("../Exp3/plots")

output_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)


# ====================================================
# EXPERIMENTAL PARAMETERS
# ====================================================

cutoff_hz = 250.0       # Low-pass cutoff frequency
filter_length = 201     # Odd length gives a centered FIR filter

# Gains
scenario = 1
gL = 1.0    # retain frequency
gM = 1.2    # modest boost
gH_1 = 1.6  # boost frequency
gH_2 = 2.0  # high boost

# ====================================================
# LOAD AND PREPROCESS THE AUDIO
# ====================================================

x, fs = sf.read(str(input_path))

# Convert stereo recordings to mono by averaging the channels.
if x.ndim == 2:
    x = np.mean(x, axis=1)

x = np.asarray(x, dtype=float)

if len(x) == 0:
    raise ValueError("The input audio file is empty.")

N = len(x)
duration = N / fs
time = np.arange(N) / fs

print("Sampling frequency:", fs, "Hz")
print("Number of samples:", N)
print("Duration:", round(duration, 3), "seconds")
print("Input peak amplitude:", float(np.max(np.abs(x))))


# ====================================================
# MANUAL CONVOLUTION
# ====================================================

def manual_convolution(signal, impulse_response):
    """Compute linear convolution using scaled, shifted copies of the signal."""
    signal = np.asarray(signal, dtype=float)
    impulse_response = np.asarray(impulse_response, dtype=float)

    y = np.zeros(len(signal) + len(impulse_response) - 1, dtype=float)

    for k in range(len(impulse_response)):
        y[k:k + len(signal)] += impulse_response[k] * signal

    return y


# ====================================================
# IMPLEMENT THE LOW-PASS FILTER
# ====================================================

def create_lowpass_filter(cutoff, sampling_frequency, length):
    """
    Create a windowed-sinc FIR low-pass filter.

    The ideal low-pass impulse response is multiplied by a Hann window.
    The coefficients are normalized so the filter has approximately unit
    gain at DC.
    """
    if length < 3 or length % 2 == 0:
        raise ValueError("Filter length must be an odd integer greater than 2.")
    if not 0 < cutoff < sampling_frequency / 2:
        raise ValueError("Cutoff must be between 0 Hz and the Nyquist frequency.")

    n = np.arange(length) - (length - 1) / 2
    normalized_cutoff = cutoff / sampling_frequency

    h = 2 * normalized_cutoff * np.sinc(2 * normalized_cutoff * n)

    window = 0.5 - 0.5 * np.cos(
        2 * np.pi * np.arange(length) / (length - 1)
    )
    h = h * window

    coefficient_sum = np.sum(h)
    if abs(coefficient_sum) < 1e-12:
        raise ValueError("Low-pass filter coefficients cannot be normalized.")

    return h / coefficient_sum


def split_frequency_bands(signal, impulse_response):
    """
    Obtain the low-frequency part by filtering.
    The FIR filter has a group delay of (M-1)/2 samples; remove that delay
    so the low-frequency component aligns with the original signal.
    The high-frequency part is then obtained by subtraction.
    """
    filtered = manual_convolution(signal, impulse_response)
    delay = (len(impulse_response) - 1) // 2
    low_frequency = filtered[delay:delay + len(signal)]
    high_frequency = signal - low_frequency

    return low_frequency, high_frequency


# ====================================================
# FOUR-MODE EQUALIZER AND MULTIPLEXER
# ====================================================

def equalizer(signal, low_frequency, high_frequency, mode):
    """Select the requested output using a mode-selection variable."""
    if mode == 1:
        return signal.copy()
    elif mode == 2:
        return gH_2 * low_frequency + gM * high_frequency
    elif mode == 3:
        return gL * low_frequency + gH_1 * high_frequency
    elif mode == 4:
        # Combined preset: low-frequency drum boost and high-frequency guitar boost.
        return gH_2 * low_frequency + gH_1 * high_frequency
    else:
        raise ValueError("Mode must be 1, 2, 3, or 4.")


# ====================================================
# MANUAL FFT IMPLEMENTATION
# Radix-2 Cooley-Tukey algorithm
# ====================================================

def fft(signal):
    """Compute the forward FFT; the input length must be a power of two."""
    N_fft = len(signal)

    if N_fft == 0:
        return []
    if N_fft & (N_fft - 1):
        raise ValueError("FFT input length must be a power of two.")
    if N_fft == 1:
        return [complex(signal[0], 0.0)]

    X = [complex(value, 0.0) for value in signal]

    # Bit-reversal permutation.
    j = 0
    for i in range(1, N_fft):
        bit = N_fft >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit

        if i < j:
            X[i], X[j] = X[j], X[i]

    # Iterative Cooley-Tukey butterfly stages.
    length = 2
    while length <= N_fft:
        half = length // 2
        angle = -2.0 * math.pi / length
        w_real = math.cos(angle)
        w_imag = math.sin(angle)

        for start in range(0, N_fft, length):
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


def fft_magnitude(signal, sampling_frequency, max_samples=262144):
    """
    Calculate a single-sided magnitude spectrum using the manual FFT.
    For long recordings, use the first segment of at most max_samples
    samples to keep the calculation and plots manageable.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 2:
        return np.array([]), np.array([])

    segment_length = min(len(signal), max_samples)
    segment_length = 1 << (segment_length.bit_length() - 1)
    segment = signal[:segment_length].copy()

    # Remove DC offset before calculating the spectrum.
    segment -= np.mean(segment)
    spectrum = fft(segment)

    magnitudes = np.abs(spectrum[:segment_length // 2 + 1]) / segment_length
    if len(magnitudes) > 2:
        magnitudes[1:-1] *= 2.0

    frequencies = (
        np.arange(segment_length // 2 + 1)
        * sampling_frequency / segment_length
    )

    return frequencies, magnitudes


# ====================================================
# PLOTTING AND SAVING HELPERS
# ====================================================

def save_audio(signal, sampling_frequency, filename):
    """Save audio as 16-bit PCM WAV, scaling only if the signal would clip."""
    signal = np.asarray(signal, dtype=float)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0

    if peak > 1.0:
        print(filename, "exceeds the [-1, 1] range; scaling to avoid clipping.")
        signal_to_save = signal / peak
    else:
        signal_to_save = signal

    output_path = output_dir / filename
    sf.write(str(output_path), signal_to_save, sampling_frequency, subtype="PCM_16")
    print("Saved audio:", output_path)


def plot_waveform_and_spectrum(original, processed, sampling_frequency,
                               original_name, processed_name, filename):
    """Plot original and processed waveforms plus their magnitude spectra."""
    original_time = np.arange(len(original)) / sampling_frequency
    processed_time = np.arange(len(processed)) / sampling_frequency

    # Use a shared waveform scale for a fair visual comparison.
    peak = max(
        float(np.max(np.abs(original))) if len(original) else 0.0,
        float(np.max(np.abs(processed))) if len(processed) else 0.0,
        1e-12
    )

    fig = plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(original_time, original)
    plt.ylim(-peak, peak)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(original_name)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(processed_time, processed)
    plt.ylim(-peak, peak)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(processed_name)
    plt.grid(True)

    plt.subplot(3, 1, 3)
    f_original, mag_original = fft_magnitude(original, sampling_frequency)
    f_processed, mag_processed = fft_magnitude(processed, sampling_frequency)

    # Avoid plotting DC on a logarithmic frequency axis.
    if len(f_original) > 1:
        plt.semilogx(
            f_original[1:], 20 * np.log10(mag_original[1:] + 1e-10),
            label=original_name
        )
    if len(f_processed) > 1:
        plt.semilogx(
            f_processed[1:], 20 * np.log10(mag_processed[1:] + 1e-10),
            label=processed_name
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Magnitude Spectrum")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plot_path = plot_dir / filename
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print("Saved plot:", plot_path)


# ====================================================
# RUN EXPERIMENT 3
# ====================================================

print("\nCreating low-pass filter...")
h = create_lowpass_filter(cutoff_hz, fs, filter_length)
print("Cutoff frequency:", cutoff_hz, "Hz")
print("Filter length:", filter_length, "samples")

print("Splitting audio into low- and high-frequency components...")
x_L, x_H = split_frequency_bands(x, h)

# Save a plot of the original waveform.
plt.figure(figsize=(12, 4))
plt.plot(time, x)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Original Audio")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir / "original_waveform.png", dpi=300)
plt.close()

mode_names = {
    1: "original",
    2: "drum_enhance",
    3: "guitar_enhance",
    4: "drum_guitar_enhance",
}

print("\nProcessing the four equalizer modes...")
for mode in [1, 2, 3, 4]:
    y = equalizer(x, x_L, x_H, mode)
    name = mode_names[mode]

    print(
        f"Mode {mode}: {name}; output samples = {len(y)}, "
        f"duration = {len(y) / fs:.3f} seconds, "
        f"peak amplitude = {np.max(np.abs(y)):.4f}"
    )

    if mode == 1:
        # Save an explicitly labelled original file as well.
        save_audio(y, fs, "original.wav")
        continue

    save_audio(y, fs, f"{name}{scenario}.wav")
    plot_waveform_and_spectrum(
        x, y, fs,
        "Original Audio", name.replace("_", " ").title(),
        f"{name}_comparison{scenario}.png"
    )

print("\nExperiment 3 completed.")
print("Audio outputs folder:", output_dir.resolve())
print("Plot folder:", plot_dir.resolve())

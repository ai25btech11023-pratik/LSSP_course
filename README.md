# Simple Audio Signal Processing

## Course project for EE1206/EE3900: Linear Systems and Signal Processing.

This project explores basic digital signal processing (DSP) techniques
on a guitar or string recording. It focuses on implementing the processing steps,
listening to the processed audio, and comparing waveforms and magnitude
spectra.


## Folder Structure

``` text
LSSP_course/
├── README.md
│── Exp1/                  # Experiment 1: LTI reverberation
│   ├── exp1.py
│   ├── Audio_outputs/
│   └──plots/                  
├── Exp2/                  # Experiment 2: LTV time-varying delay effect
│   ├── exp2.py
│   ├── Audio_outputs/
│   └── plots/                 
├── Exp3/                  # Experiment 3: two-mode / multi-mode equalizer
│   ├── exp3.py
│   ├── Audio_outputs/
│   └── plots/                
├── recordings/            # Input audio recording
│   └── LP_Lick10_FN.wav   
└── requirements.txt       # Python dependencies
```

## Experiments

### Experiment 1 --- Room Reverberation using an LTI System

**Goal:** Simulate room reflections by passing a guitar recording
through a linear time-invariant (LTI) system.

The output is formed by convolving the input signal (x\[n\]) with an
impulse response (h\[n\]):

$$ y\[n\] = x\[n\] \* h\[n\] $$

The impulse response contains a direct sound and several delayed,
attenuated reflections:

$$ h[n] = \delta[n] + \sum_{i=1}^{M} a_i \delta[n-d_i], $$

where $a_i$ is the amplitude of reflection $i$, and $d_i$ is its delay
in samples.

**Processing steps**

1. Load the guitar recording and obtain its sampling frequency.
2. Construct the impulse response with a direct impulse and delayed reflections.
3. Convolve the recording with the impulse response.
4. Save or play the reverberant output.
5. Plot the impulse response and compare the original and processed waveforms and magnitude spectra.

**Parameters to report**

- **Input audio file and sampling frequency:** Set from the selected recording.
- **Reflection amplitudes:** The values of \(a_i\) defined in `exp1.py`.
- **Reflection delays:** The values of \(d_i\) defined in `exp1.py`. Check whether the code specifies them in seconds or samples.
- **Number of reflections:** \(M\), the number of delayed reflection taps, excluding the direct impulse.
- **Spectrum display:** State whether the plot uses linear magnitude or magnitude in decibels. For a magnitude spectrum expressed in dB, use:

$$
X_{\mathrm{dB}}[k] = 20\log_{10}\left(|X[k]|+\epsilon\right)
$$

**Important:** The exact reflection amplitudes and delays depend on the values currently configured in your script. Copy those values into this section before submitting the README; do not assume example parameter values from other implementations apply to your code.

### Experiment 2 — Musical Effect using an LTV System

**Goal:** Create a time-varying delay effect. Unlike a fixed-delay echo, the delay changes periodically over time.

A typical model is:

$$
y[n] = x[n] + \alpha x[n-D[n]]
$$

The time-varying delay is:

$$
D[n] = D_0 + \Delta D \sin\left(\frac{2\pi f_m n}{f_s}\right)
$$

where:

- \(\alpha\) is the delayed-signal mix/gain.
- \(D_0\) is the nominal delay.
- \(\Delta D\) is the delay variation.
- \(f_m\) is the modulation frequency.
- \(f_s\) is the sampling frequency.

**Processing steps**

1. Load the same guitar recording used for Experiment 1.
2. Calculate the periodically varying delay for each sample.
3. Add the delayed sample, scaled by \(\alpha\), to the original signal.
4. Listen to the processed recording.
5. Compare the original and processed waveforms and magnitude spectra, and experiment with modulation frequency and delay variation.

**Parameters to report**

- \(\alpha\): Delayed-signal gain/mix in `task2.py`.
- \(D_0\): Nominal delay, in the units used by the code.
- \(\Delta D\): Delay variation, in the units used by the code.
- \(f_m\): Modulation frequency in Hz.
- \(f_s\): Sampling frequency of the input audio.
- **Output duration:** Process the full input recording if that is the intended comparison.

Copy the actual values from the parameter definitions in `task2.py` into the report. If the implementation uses a fixed number of samples for delay, state that the delay is in samples; if it converts seconds to samples, state both the time and sample values.

## Libraries and Their Uses

The exact dependencies are determined by the imports in your Python files. Common libraries for this project are:

| Library / module | Usage |
|---|---|
| `math` | Mathematical operations such as sine, cosine, and \(\pi\), including delay modulation and manual DFT/FFT calculations. |
| `wave` | Reading and writing WAV audio files when the code uses Python's built-in WAV support. |
| `struct` | Converting raw WAV byte data to and from numerical sample values, if used by the audio I/O code. |
| `os` / `pathlib` | Building file paths and creating output directories, if used. |
| `matplotlib` | Plotting waveforms, impulse responses, and magnitude spectra. |
| `numpy` | Array operations and numerical processing, only if imported by the implementation. |
| `soundfile` | Reading and writing audio files, only if imported by the implementation. |
| `scipy` | Signal-processing or audio utilities, only if imported by the implementation. |

The manual DFT/FFT implementation does not require NumPy for the transform itself. Keep only the libraries that are actually imported by your submitted scripts in the final dependency list.

## Requirements

Use Python 3.10 or newer if that matches your environment. Install only the packages imported by the scripts. For example, if the project uses Matplotlib and SoundFile:

```bash
python -m pip install matplotlib soundfile
```

If NumPy or SciPy is imported, install those as well:

```bash
python -m pip install numpy scipy
```

Do not install `wave`, `struct`, `math`, or `os` separately; they are part of Python's standard library.

## Running the Code

1. Place the input WAV recording in the expected data folder, or update the input path in each script.
2. Open a terminal in the project directory.
3. Install the dependencies required by your imports.
4. Run Experiment 1:

   ```bash
   python exp1.py
   ```

5. Run Experiment 2:

   ```bash
   python task2.py
   ```

6. Run Experiment 3 separately when needed (listed here only in the folder structure):

   ```bash
   python exp3.py
   ```

If your files have different names, substitute the actual filenames. For example, run `python "exp1(2).py"` if that is the filename on disk.

## Input Dataset and Audio

Record the following information for reproducibility:

- **Dataset name:** [Enter the dataset name]
- **Dataset URL:** [Enter the Kaggle dataset URL]
- **Selected audio file:** [Enter the WAV filename]
- **Sampling frequency:** [Enter the sample rate in Hz]
- **Audio license / usage conditions:** [Enter the applicable license information]

Use the same recording for Experiments 1 and 2 when comparing their effects, as required by the project brief. Ensure that the sampling frequency used by the code is the actual sample rate of the recording.

## Outputs

Depending on how the scripts are configured, save or generate:

- **Experiment 1:** Impulse-response plot, original/reverberant waveform comparison, magnitude-spectrum comparison, and reverberant WAV.
- **Experiment 2:** Original/processed waveform comparison, magnitude-spectrum comparison, and time-varying-delay WAV.

Label plots and output audio files clearly so the original and processed versions can be distinguished.

## Notes on Magnitude Spectra

If a spectrum is calculated as the magnitude \(|X[k]|\), it is a **linear magnitude spectrum**, not a decibel spectrum. To convert magnitude to decibels, use:

$$
X_{\mathrm{dB}}[k] = 20\log_{10}\left(|X[k]|+\epsilon\right)
$$

Here, \(\epsilon\) is a small positive number that prevents taking the logarithm of zero. Use the same scaling and frequency range when comparing the original and processed spectra.

## Final Checklist

- [ ] Confirm the input WAV file path and sample rate.
- [ ] Replace parameter descriptions with the actual values used in each script.
- [ ] Confirm whether plotted spectra are linear magnitude or dB.
- [ ] Ensure output plots and audio files are saved with clear filenames.
- [ ] Keep the dependency list consistent with the imports in the submitted code.
- [ ] Add the dataset name, URL, and license information.

## Academic Integrity

Be prepared to explain the impulse response, convolution, time-varying
delay, and the effect of changing each parameter. Acknowledge AI
assistance according to the course instructions and follow the dataset's
license terms.

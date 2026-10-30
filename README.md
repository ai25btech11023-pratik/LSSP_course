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
- **Reflection delays:** The values of \(d_i\) defined in `exp1.py`.
- **Number of reflections:** \(M\), the number of delayed reflection taps, excluding the direct impulse.
- **Spectrum display:** the plot uses linear magnitude for spectrum plot against frequency.
- 
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

- $\alpha$ is the delayed-signal mix/gain.
- $D_0$ is the nominal delay.
- $\Delta D$ is the delay variation.
- $f_m$ is the modulation frequency.
- $f_s$ is the sampling frequency.

**Processing steps**

1. Load the same guitar recording used for Experiment 1.
2. Calculate the periodically varying delay for each sample.
3. Add the delayed sample, scaled by $\alpha$, to the original signal.
4. Listen to the processed recording.
5. Compare the original and processed waveforms and magnitude spectra, and experiment with modulation frequency and delay variation.

**Parameters to report**

- $\alpha$: Delayed-signal gain/mix in `exp2.py`.
- $D_0$ : Nominal delay in ms in `exp2.py`.
- $\Delta D$: Delay variation in ms in `exp2.py`.
- $f_m$: Modulation frequency in Hz.
- $f_s$: Sampling frequency of the input audio.

## Libraries and Their Uses

Common libraries for this project are:

| Library / module | Usage |
|---|---|
| `math` | Mathematical operations such as sine, cosine, and \(\pi\), including delay modulation and manual DFT/FFT calculations. |
| `soundfile` | Reading and writing WAV audio files|
| `pathlib` | Building file paths and creating output directories|
| `matplotlib` | Plotting waveforms, impulse responses, and magnitude spectra. |
| `numpy` | Array operations and numerical processing, only if imported by the implementation. |

## Requirements

```bash
python -m pip install numpy matplotlib soundfile pathlib
```


## Running the Code

1. Place the input WAV recording in the recordings directory.
2. Open a terminal in the LSSP_course directory.
3. Install the dependencies required by your imports.
4. Go to desired directory for experiment to be run.
5. for instance  Experiment 1:

   ```bash
   cd Exp1
   python exp1.py
   ```

6. Run Experiment 2:

   ```bash
   cd Exp2
   python task2.py
   ```

7. Run Experiment 3:

   ```bash
   cd Exp3
   python exp3.py
   ```

## Input Dataset and Audio


- **Dataset name:** Musical instrument detection
- **Dataset URL:** https://www.kaggle.com/code/dima806/musical-instrument-detection/input
- **Selected audio file:** LP_Lick10_FN.wav (guitar), 2_ray-chen_bwv1004_mov5.wav (violin)
- **Sampling frequency:** 44.1 kHz
- **Audio license / usage conditions:** This Dataset is collected From some scraping from few sites & some open source datasets.

## Outputs

- **Experiment 1:** Impulse-response plot, original/reverberant waveform comparison, magnitude-spectrum comparison, and reverberant WAV.
- **Experiment 2:** Original/processed waveform comparison, magnitude-spectrum comparison, and time-varying-delay WAV.

## Academic Integrity

AI has been used to assist in minor error correction with regards to syntax and efficiency only. 

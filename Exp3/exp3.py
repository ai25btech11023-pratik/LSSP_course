import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pathlib import Path


def lowpass(x,alpha):
    low= np.empty_like(x)
    low[0]=x[0]

    for n in range(1 , len(x)):

        low[n]=alpha*x[n]+ (1-alpha)*low[n-1]

    return low


def fft(x):
    n=len(x)
    if n== 1:
        return x

    even= fft(x[::2])

    odd= fft(x[1::2])
    w=np.exp(-2j*np.pi* np.arange(n//2)/n)

    return np.concatenate((even+w*odd, even-w* odd))



def spectrum(x,fs):
    n=1
    while n <len(x):
        n= n*2

    padded=np.zeros(n)
    padded[:len(x)] = x

    magnitude= np.abs(fft(padded)[:n//2])
    frequency= np.arange(n//2)*fs/n

    db = 20* np.log10(magnitude + 1e-10)

    return frequency,db


def plot_comparison(x,y,fs, name,path):

    t= np.arange(len(x))/fs
    peak= max(np.max(np.abs(x)), np.max(np.abs(y)), 0.01)
    # use one second from the middle for FFT comparison
    start= max(0,(len(x)-fs)//2)
    end= min(len(x),start+fs)

    f,original_db= spectrum(x[start:end],fs)
    processed_db= spectrum(y[start:end],fs)[1]

    fig, ax=plt.subplots(3, 1, figsize=(10, 8))

    ax[0].plot(t,x)
    ax[0].set_title("Original")

    ax[1].plot(t,y,color="orange")
    ax[1].set_title(name)

    for a in ax[:2]:
     
        a.set_ylim(-peak,peak)

        a.set_xlabel("Time(secs)")
        a.set_ylabel("Amplitude")

    ax[2].plot(f,original_db, label="Original")
    ax[2].plot(f,processed_db,label=name)
    ax[2].set_xlim(0,3000)

    ax[2].set_title("Magnitude spectrum")
    ax[2].set_xlabel("Frequency (Hz)")
    ax[2].set_ylabel("Magnitude(dB)")

    ax[2].legend()
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    folder = Path(__file__).resolve().parent
    input_path = folder.parent / "recordings" / "exp3_audio.wav"
    audio_dir = folder / "Audio_outputs"
    plot_dir = folder / "plots"

    fc = 250  # cutoff in Hz
    
    modes = [
    ("exp3_mode2_drum_enhance","drum enhance", 2.0, 1.0),
    ("exp3_mode3_guitar_enhance","guitar enhance",1.0, 1.6),
    ("exp3_mode4_drum_guitar_enhance","drum guitar enhance",2.0, 1.6),
    ]


    x, fs = sf.read(str(input_path))
    if x.ndim == 2:
        x =x.mean(axis=1)
    
    
    # leave room for the gain before processing
    peak = np.max(np.abs(x))
    if peak > 0:
        x = 0.5*x/peak

    alpha= 2*np.pi*fc/(fs + 2*np.pi*fc)
    low= lowpass(x, alpha)
    high= x - low

    audio_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    print("Sampling rate:",fs, "Hz")
    print("Cutoff:", fc, "Hz, alpha =", round(alpha, 4))

    
    for filename, title, gain_low, gain_high in modes:
        y= gain_low*low + gain_high*high

        peak = np.max(np.abs(y))
        if peak >1:
            y= y/peak

        sf.write(audio_dir / (filename + ".wav"), y,fs, subtype="PCM_16")
        plot_comparison(x,y,fs, title,
                        plot_dir/(filename + "_comparison.png"))
        print("saved",filename)

main()

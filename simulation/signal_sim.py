"""
Simulate signals for testing communication across frequency bands.

Two signals s_a and s_b
s_a drives s_b
if s_b is at an excitable state and receives a high-gamma burst
    s_b generates a high-gamma burst.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from acoustics.generator import white, pink

FSAMPLE = 1000.

def sim(dur):
    """
    Arguments
    dur: Duration (in s)

    Returns
    t: Vector of time-points (in s)
    s_a: Signal at area a
    s_b: Signal at area b
    """

    n_samps = int(dur * FSAMPLE)
    t = np.arange(n_samps) / FSAMPLE

    #######################################
    # Simulate a source signal in which   #
    # alpha pulses inhibit gamma activity #
    #######################################

    # Make a low-freq signal with fluctuating frequency
    a_alpha_sig = osc_var_freq(n_samps, fs=FSAMPLE, low=6, high=14, speed=0.1)

    # Make a high-gamma signal that depends on alpha amplitude
    # Following Jiang et al 2015, NeuroImage
    #    This implement a strict inhibitory role of alpha. Gamma just chugs
    #    along until it is inhibited by an alpha-pulse.
    gamma_freq = (60, 200) # Hz
    a_gamma_scale = 0.5 # How strong is the gamma activity
    a_lf_gamma_lag = 0.0 # Lag the gamma signal (in s) from the LF trough
    a = 10 # Sigmoid slope
    c = 0.1 # Sigmoid "threshold"
    sigmoid = lambda x,a,c: 1 - (1 / (1 + np.exp(-a * (x - c))))
    a_gamma_osc = offset_bp_noise(n_samps, *gamma_freq)
    a_gamma_sig = sigmoid(a_alpha_sig, a, c) * a_gamma_osc
    a_gamma_sig = np.roll(a_gamma_sig, # Gamma is at a lag from alpha
                          int(a_lf_gamma_lag * FSAMPLE)) 
    a_gamma_sig = a_gamma_sig * a_gamma_scale

    # Add white noise and pink noise
    white_amp = 0.001
    pink_amp = 0.001
    white_noise = white_amp * normalize(white(n_samps))
    pink_noise = pink_amp * normalize(pink(n_samps))

    # Combine LF and HF components
    s_a = a_alpha_sig + a_gamma_sig + white_noise + pink_noise
    
    ################################
    # Simulate a desination signal #
    ################################

    # Make an alpha signal with fluctuating amplitude
    # This controls excitability in area b.
    b_alpha_sig = osc_var_freq(n_samps, fs=FSAMPLE, low=6, high=14, speed=0.1)

    # Gamma activity in s_a triggers activity in s_b when are b is in an
    # excitable state. Low alpha leads to excitable state
    a_b_lag = 0.015 # Lag between activity in s_a and s_b
    b_gamma_scale = 2.0
    b_gamma_inp = np.roll(a_gamma_sig, int(a_b_lag * FSAMPLE)) # Gamma input
    b_gamma_sig = sigmoid(b_alpha_sig, a, c) * b_gamma_inp * b_gamma_scale

    # Add white noise and pink noise
    white_noise = white_amp * normalize(white(n_samps))
    pink_noise = pink_amp * normalize(pink(n_samps))

    # Combine LF and HF components
    s_b = b_alpha_sig + b_gamma_sig + white_noise + pink_noise

    return t, s_a, s_b


def plot_signals(t, s_a, s_b):
    plt.clf()
    plt.plot(t, s_a)
    plt.plot(t, s_b)
    plt.xlim(0, 2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


#############################
# Signal processing helpers #
#############################

def normalize(x):
    return x / np.max(x)


def bp_noise(n_samps, low, high):
    """ Band-limited noise
    """
    band_lim_noise = bp_filter(white(n_samps), low, high, FSAMPLE)
    return band_lim_noise


def offset_bp_noise(n_samps, low, high):
    """ Band-limited noise that is strictly positive
    """
    return bp_noise(n_samps, low / 2, high / 2) ** 2


def bp_filter(data, lowcut, highcut, fs, order=2):
    def butter_bandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def osc_var_freq(n_samps, fs, low, high, speed):
    """ An oscillation with variable frequency

    n_samps: Number of samples in the output
    fs: Sampling rate (Hz)
    low: Lower frequency for the oscillations
    high: Upper frequency for the oscillations
    speed: How quickly the frequency can change
    """

    f = bounded_walk(n_samps, low, high, speed) # instantaneous freq
    phi = np.cumsum(2 * np.pi * f / fs) # Change in phase per sample
    x = np.sin(phi)
    x = (1 / 2) * (x + 1) # Make it vary from 0 to 1
    return x


def bounded_walk(n_samps, low, high, speed):
    """ A random walk that is bounded within [low, high], changing at 'speed'
    """
    x_steps = np.random.normal(scale=speed, size=n_samps)
    x = [np.mean([low, high])] # Initialize at the mean value
    for e in x_steps:
        next_val = x[-1] + e
        if (next_val < low) or (next_val > high):
            next_val = x[-1] - e
        x.append(next_val)
    x.pop(0)
    return np.array(x)


###########################
# Run a simple simulation #
###########################

if __name__ == '__main__':
    t, s_a, s_b = sim(10)
    plot_signals(t, s_a, s_b)

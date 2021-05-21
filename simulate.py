"""
Simulate signals for testing communication across frequency bands.

Two signals s_a and s_b
s_a drives s_b
if s_b is at an excitable state and receives a high-gamma burst
    s_b generates a high-gamma burst.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from acoustics.generator import white, pink

plot_dir = '../data/plots/simulated/illustrate/'

alpha_freq = (9, 11)  # Hz
gamma_freq = (70, 100)


def sim(dur=10, fs=1000, noise_amp=0.1,
        alpha_freq=alpha_freq,
        gamma_freq=gamma_freq,
        signal_leakage=0.0,
        common_noise_amp=0.1, common_alpha_amp=0.0,
        gamma_lag_a=0.015, gamma_lag_a_to_b=0.015,
        shared_gamma=True,
        plot=False):
    """
    Simulate phase-locked communication across frequency bands.

    Parameters
    ----------
    dur : float
        Duration (in s)
    fs : float
        Sampling rate (Hz)
    noise_amp : float
        Amplitude of the noise added to the signals
    signal_leakage: float (0.0 - 1.0)
        How much does each signal show up in the other channel?
    common_noise_amp: float
        Amplitude of the noise that's added to both signals
    common_alpha_amp: float
        Amplitude of the alpha signal that's added to both signals
    gamma_lag_a : float
        Duration by which gamma activity is lagged from the LF trough w/in s_a
    gamma_lag_a_to_b : float
        Duration by which gamma activity in s_a is lagged in s_b
    shared_gamma : bool
        Is the gamma signal shared between regions (True) or does each region
        have an independently-generated gamma signal (False)?

    Returns
    -------
    t : np.ndarray
        Vector of time-points (in s)
    s_a : np.ndarray
        Signal at area a
    s_b : np.ndarray
        Signal at area b
    """

    n_samps = int(dur * fs)
    t = np.arange(n_samps) / fs

    #######################################
    # Simulate a source signal in which   #
    # alpha pulses inhibit gamma activity  #
    #######################################

    # Make a low-freq signal with fluctuating frequency
    a_alpha_sig = osc_var_freq(n_samps, fs=fs,
                               low=alpha_freq[0], high=alpha_freq[1],
                               speed=0.1)

    if plot:  # Plot some example alpha activity
        n_samps_to_plot = int(fs / 2)
        plt.figure(figsize=(3, 2))
        plt.plot(a_alpha_sig[:n_samps_to_plot])
        plt.axis('off')
        plt.savefig(f"{plot_dir}alpha.png")

    # Make a high-gamma signal that depends on alpha amplitude
    # Following Jiang et al 2015, NeuroImage
    #    This implement a strict inhibitory role of alpha. Gamma just chugs
    #    along until it is inhibited by an alpha-pulse.
    a_gamma_scale = 1  # How strong is the gamma activity
    a = 10  # Sigmoid slope
    c = 0.1  # Sigmoid "threshold"
    a_gamma_osc = osc_var_freq(n_samps, fs, gamma_freq[0], gamma_freq[1], 0.5)
    a_gamma_osc -= 0.5

    if plot:  # Plot some example gamma activity
        plt.clf()
        plt.plot(a_gamma_osc[:n_samps_to_plot])
        plt.axis('off')
        plt.savefig(f"{plot_dir}gamma.png")

        # Plot the sigmoid
        x_sigspace = np.linspace(-1, 1, 100)
        x_sigmoid = sigmoid(x_sigspace, a, c)
        plt.clf()
        plt.plot(x_sigspace, x_sigmoid)
        plt.axis('on')
        plt.xticks([-1, 0, 1])
        plt.yticks([0, 1])
        plt.xlabel('Alpha signal')
        plt.ylabel('HF amplitude')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}sigmoid.png")

    # Multiply the gamma activity by the sigmoid
    a_gamma_sig = sigmoid(a_alpha_sig, a, c) * a_gamma_osc
    a_gamma_sig = np.roll(a_gamma_sig,  # Gamma is at a lag from alpha
                          int(gamma_lag_a * fs))
    a_gamma_sig = a_gamma_sig * a_gamma_scale

    if plot:  # Plot the gamma activity with PAC
        plt.clf()
        plt.plot(a_gamma_sig[:n_samps_to_plot])
        plt.axis('off')
        plt.savefig(f"{plot_dir}gamma_PAC.png")

    # Add noise
    noise = noise_amp * normalize(pink(n_samps))

    # Combine LF and HF components (and noise)
    s_a = a_alpha_sig + a_gamma_sig + noise

    if plot:  # Plot the resulting signal
        plt.clf()
        plt.plot(s_a[:n_samps_to_plot])
        plt.axis('off')
        plt.savefig(f"{plot_dir}s_a_sender.png")

    ################################
    # Simulate a desination signal  #
    ################################

    # Make an alpha signal with fluctuating amplitude
    # This controls excitability in area b.
    b_alpha_sig = osc_var_freq(n_samps, fs=fs, low=6, high=14, speed=0.1)

    # Gamma activity in s_a triggers activity in s_b when are b is in an
    # excitable state. Low alpha leads to excitable state
    if shared_gamma:  # Propagate the gamma from the sender
        b_gamma_sig = np.roll(a_gamma_sig, int(gamma_lag_a_to_b * fs))
    else:  # Generate independent gamma
        b_gamma_osc = osc_var_freq(n_samps, fs,
                                   gamma_freq[0], gamma_freq[1],
                                   0.5) - 0.5
        b_gamma_sig = np.roll(b_gamma_osc, int(gamma_lag_a_to_b * fs))

    b_gamma_scale = 1.0
    b_gamma_sig = sigmoid(b_alpha_sig, a, c) * b_gamma_sig * b_gamma_scale

    if plot:  # Plot the gamma signal in the receiver
        plt.clf()
        plt.plot(b_gamma_sig[:n_samps_to_plot])
        plt.axis('off')
        plt.savefig(f"{plot_dir}gamma_b.png")

    # Add noise
    noise = noise_amp * normalize(pink(n_samps))

    # Combine LF and HF components
    s_b = b_alpha_sig + b_gamma_sig + noise

    if plot:  # Plot the resulting signal
        plt.clf()
        plt.plot(s_b[:n_samps_to_plot])
        plt.axis('off')
        plt.savefig(f"{plot_dir}s_b_receiver.png")

    #############################
    # Add noise to both signals  #
    #############################

    # Add each signal to the other, simulating volume conduction
    s_a, s_b = (s_a + (s_b * signal_leakage)), ((s_a * signal_leakage) + s_b)

    # Add common noise to both signals
    noise = common_noise_amp * normalize(pink(n_samps))
    s_a = s_a + noise
    s_b = s_b + noise

    # Add common LF oscillations to both signals, simulating volume conduction
    common_alpha_sig = osc_var_freq(n_samps, fs=fs, low=6, high=14, speed=0.1)
    common_alpha_sig = common_alpha_sig * common_alpha_amp
    s_a = s_a + common_alpha_sig
    s_b = s_b + common_alpha_sig

    return t, s_a, s_b


def sim_lf_coh_plus_noise(dur, fs, lag=0, noise_amp=1, osc_amp=1):
    """
    Sig A: Alpha oscillation plus pink noise
    Sig B: Same alpha oscillation as Sig A plus independent pink noise
    Result: Alpha-limited phase-dependent communication b/w A & B, with
    directionality haphazard between HF frequencies
    Solution: Shuffling HF info across trials/epochs will lead to a permuted
    distribution that has similar levels of phase-diff-TE. But when there's
    real communication, it will be stronger in the shuffled case.
    """
    n = int(dur * fs)
    t = np.arange(n) / fs
    s_osc = osc_amp * osc_var_freq(n, fs,
                                   alpha_freq[0], alpha_freq[1],
                                   0.1)
    s_a = s_osc + (noise_amp * pink(n))
    s_b = s_osc + (noise_amp * pink(n))
    s_b = np.roll(s_b, lag)
    return t, s_a, s_b


def sim_lf_coh_with_pac(dur, fs, lag,
                        noise_amp=1.5, osc_amp=1, gamma_amp=1,
                        alpha_freq=alpha_freq, gamma_freq=gamma_freq):
    """
    Two signals with LF coherence and independent PAC
    """
    n = int(dur * fs)
    t = np.arange(n) / fs
    lf_osc = osc_amp * osc_var_freq(n, fs,
                                    alpha_freq[0], alpha_freq[1],
                                    0.1)
    a = 10  # Sigmoid slope
    c = 0.1  # Sigmoid "threshold"

    def gamma():
        return osc_var_freq(n, fs, gamma_freq[0], gamma_freq[1], 0.5) - 0.5

    a_gamma = sigmoid(lf_osc, a, c) * gamma()
    b_gamma = sigmoid(lf_osc, a, c) * gamma()
    s_a = lf_osc + (gamma_amp * a_gamma) + (noise_amp * pink(n))
    s_b = lf_osc + (gamma_amp * b_gamma) + (noise_amp * pink(n))
    s_b = np.roll(s_b, lag)
    return t, s_a, s_b


def sim_lf_coh_with_hf_comm(dur, fs, lag, noise_amp=1.5,
                            osc_amp=1, gamma_amp=1):
    """
    Two signals with LF coherence and HF communication, but the HF
    communication *does not* depend on the LF phase difference.
    """
    t, s_a, s_b = sim_lf_coh_plus_noise(dur, fs, lag, noise_amp, osc_amp)
    gamma_sig = osc_var_freq(len(t), fs, gamma_freq[0], gamma_freq[1], 0.5)
    gamma_sig -= 0.5
    gamma_sig *= gamma_amp
    s_a += gamma_sig
    s_b += np.roll(gamma_sig, lag)
    return t, s_a, s_b


#############################
# Signal processing helpers  #
#############################

def sigmoid(x, a, c):
    return 1 - (1 / (1 + np.exp(-a * (x - c))))


def normalize(x):
    return x / np.max(x)


def bp_noise(n_samps, fs, low, high):
    """ Band-limited noise
    """
    band_lim_noise = bp_filter(white(n_samps), low, high, fs)
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
    y = filtfilt(b, a, data)
    return y


def osc_var_freq(n_samps, fs, low, high, speed):
    """ An oscillation with variable frequency

    Args:
        n_samps: Number of samples in the output
        fs: Sampling rate (Hz)
        low: Lower frequency for the oscillations
        high: Upper frequency for the oscillations
        speed: How quickly the frequency can change

    Returns:
        x: A vector of floats
    """

    f = bounded_walk(n_samps, low, high, speed)  # instantaneous freq
    phi = np.cumsum(2 * np.pi * f / fs)  # Change in phase per sample
    x = np.sin(phi)
    x = (1 / 2) * (x + 1)  # Make it vary from 0 to 1
    return x


def bounded_walk(n_samps, low, high, speed):
    """ A random walk that is bounded within [low, high], changing at 'speed'
    """
    x_steps = np.random.normal(scale=speed, size=n_samps)
    x = [np.mean([low, high])]  # Initialize at the mean value
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
    t, s_a, s_b = sim()
    plt.clf()
    plt.plot(t, s_a)
    plt.plot(t, s_b)
    plt.xlim(0, 2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
